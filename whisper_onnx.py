#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
Please first run ./export-onnx.py
before you run this script
"""
import argparse
import base64
from typing import Tuple
import time

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import librosa

# Add openvino libs to path as onnxruntime_providers_openvino.dll depends on openvino.dll. See https://github.com/intel/onnxruntime/releases/
import onnxruntime.tools.add_openvino_win_libs as utils
utils.add_openvino_libs_to_path()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large', 'turbo'],
        help="Model type",
    )

    parser.add_argument(
        "--language",
        type=str,
        help="""The actual spoken language in the audio.
        Example values, en, de, zh, jp, fr.
        If None, we will detect the language using the first 30s of the
        input audio
        """,
    )

    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        type=str,
        default="transcribe",
        help="Valid values are: transcribe, translate",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Execution device. Use 'CPU', 'GPU', 'NPU' for OpenVINO. If not specified, CPUExecutionProvider will be used by default."
    )

    parser.add_argument(
        "sound_file",
        type=str,
        help="Path to the test wave",
    )
    return parser.parse_args()


def create_attention_mask(current_index: int, max_ctx: int = 448) -> torch.Tensor:
    attention_mask = torch.full(
        (1, 1, 1, max_ctx),
        float("-inf"),
        dtype=torch.float32
    )

    end_index = current_index + 1
    if end_index > max_ctx:
        end_index = max_ctx

    attention_mask[0, 0, 0, :end_index] = 0.0
    
    return attention_mask


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        encoder_device: str,
        decoder_device: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts
        self.encoder_device = encoder_device
        self.decoder_device = decoder_device

        self.init_encoder(encoder, encoder_device)
        self.init_decoder(decoder, decoder_device)

    def init_encoder(self, encoder: str, device: str):
        if device in ["CPU", "GPU", "NPU"]:
            print(f"Encoder device: OpenVINO EP with device = {device}")
            providers = ['OpenVINOExecutionProvider']
            provider_options = [{"device_type": device}]
            
            # For NPU device caching
            if device == "NPU":
                 provider_options[0]["cache_dir"] = "cache"
        else:
            print("Encoder device: Using Default CPU Executor.")
            providers = ["CPUExecutionProvider"]
            provider_options = None
            
        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=providers,
            provider_options=provider_options
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_text_layer = int(meta["n_text_layer"])
        self.n_text_ctx = int(meta["n_text_ctx"])
        self.n_text_state = int(meta["n_text_state"])
        self.n_mels = int(meta["n_mels"])
        self.sot = int(meta["sot"])
        self.eot = int(meta["eot"])
        self.translate = int(meta["translate"])
        self.transcribe = int(meta["transcribe"])
        self.no_timestamps = int(meta["no_timestamps"])
        self.no_speech = int(meta["no_speech"])
        self.blank = int(meta["blank_id"])

        self.sot_sequence = list(map(int, meta["sot_sequence"].split(",")))
        self.sot_sequence.append(self.no_timestamps)

        self.all_language_tokens = list(
            map(int, meta["all_language_tokens"].split(","))
        )
        self.all_language_codes = meta["all_language_codes"].split(",")
        self.lang2id = dict(zip(self.all_language_codes, self.all_language_tokens))
        self.id2lang = dict(zip(self.all_language_tokens, self.all_language_codes))

        self.is_multilingual = int(meta["is_multilingual"]) == 1

    def init_decoder(self, decoder: str, device: str):
        if device in ["CPU", "GPU", "NPU"]:
            print(f"Decoder device: OpenVINO EP with device = {device}")
            providers = ['OpenVINOExecutionProvider']
            provider_options = [{"device_type": device}]
            
            # For NPU device, OpenVINO typically benefits from caching
            if device == "NPU":
                 provider_options[0]["cache_dir"] = "cache"
        else:
            print("Decoder device: Using Default CPU Executor.")
            providers = ["CPUExecutionProvider"]
            provider_options = None
            
        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=providers,
            provider_options=provider_options
        )

    def run_encoder(
        self,
        mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        
        n_layer_cross_k, n_layer_cross_v = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: mel.numpy(),
            },
        )

        end = time.time()
        print(f"Encoder processing time: {(end - start) * 1000:.2f} ms")

        return torch.from_numpy(n_layer_cross_k), torch.from_numpy(n_layer_cross_v)

    def run_decoder(
        self,
        tokens: torch.Tensor,
        n_layer_self_k_cache: torch.Tensor,
        n_layer_self_v_cache: torch.Tensor,
        n_layer_cross_k: torch.Tensor,
        n_layer_cross_v: torch.Tensor,
        offset: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = time.time()

        logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
            ],
            {
                self.decoder.get_inputs()[0].name: tokens.numpy(),
                self.decoder.get_inputs()[1].name: n_layer_self_k_cache.numpy(),
                self.decoder.get_inputs()[2].name: n_layer_self_v_cache.numpy(),
                self.decoder.get_inputs()[3].name: n_layer_cross_k.numpy(),
                self.decoder.get_inputs()[4].name: n_layer_cross_v.numpy(),
                self.decoder.get_inputs()[5].name: offset.numpy(),             
                self.decoder.get_inputs()[6].name: attention_mask.numpy(),    
            },
        )
        
        end = time.time()
        print(f"Decoder processing time: {(end - start) * 1000:.2f} ms")

        return (
            torch.from_numpy(logits),
            torch.from_numpy(out_n_layer_self_k_cache),
            torch.from_numpy(out_n_layer_self_v_cache),
        )

    def get_self_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = 1
        n_layer_self_k_cache = torch.zeros(
            batch_size,
            self.n_text_layer,
            self.n_text_ctx,
            self.n_text_state,
        )
        n_layer_self_v_cache = torch.zeros(
            batch_size,
            self.n_text_layer,
            self.n_text_ctx,
            self.n_text_state,
        )
        return n_layer_self_k_cache, n_layer_self_v_cache

    def suppress_tokens(self, logits, is_initial: bool) -> None:
        # suppress blank
        if is_initial:
            logits[self.eot] = float("-inf")
            logits[self.blank] = float("-inf")

        # suppress <|notimestamps|>
        logits[self.no_timestamps] = float("-inf")

        logits[self.sot] = float("-inf")
        logits[self.no_speech] = float("-inf")

        # logits is changed in-place
        logits[self.translate] = float("-inf")

    def detect_language(
        self, n_layer_cross_k: torch.Tensor, n_layer_cross_v: torch.Tensor
    ) -> int:
        tokens = torch.tensor([[self.sot]], dtype=torch.int64)
        offset = torch.zeros(1, dtype=torch.int64)

        attention_mask = create_attention_mask(offset.item(), self.n_text_ctx)
        
        n_layer_self_k_cache, n_layer_self_v_cache = self.get_self_cache()

        logits, n_layer_self_k_cache, n_layer_self_v_cache = self.run_decoder( 
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
            attention_mask=attention_mask,
        )
        
        logits = logits.reshape(-1)
        mask = torch.ones(logits.shape[0], dtype=torch.int64)
        for lang_token in self.all_language_tokens:
             mask[lang_token] = 0
        logits[mask != 0] = float("-inf")
        lang_id = logits.argmax().item()
        print("detected language: ", self.id2lang[lang_id])
        return lang_id


def load_tokens(filename):
    tokens = dict()
    with open(filename, "r") as f:
        for line in f:
            t, i = line.split()
            tokens[int(i)] = t
    return tokens


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    return samples, sample_rate


def compute_features(filename: str, dim: int = 80) -> torch.Tensor:
    """
    Args:
      filename:
        Path to an audio file.
    Returns:
      Return a 1-D float32 tensor of shape (1, 80, 3000) containing the features.
    """
    wave, sample_rate = load_audio(filename)
    if sample_rate != 16000:
        # import librosa
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    features = []
    opts = knf.WhisperFeatureOptions()
    opts.dim = dim
    online_whisper_fbank = knf.OnlineWhisperFbank(opts)
    online_whisper_fbank.accept_waveform(16000, wave)
    online_whisper_fbank.input_finished()
    for i in range(online_whisper_fbank.num_frames_ready):
        f = online_whisper_fbank.get_frame(i)
        f = torch.from_numpy(f)
        features.append(f)

    features = torch.stack(features)

    log_spec = torch.clamp(features, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    mel = (log_spec + 4.0) / 4.0
    # mel shape: (T, 80)

    target_T = 3000
    current_T = mel.shape[0]

    if current_T > target_T:
        mel = mel[:target_T]
    elif current_T < target_T:
        padding = target_T - current_T
        mel = torch.nn.functional.pad(mel, (0, 0, 0, padding), "constant", 0)
    
    # mel shape: (3000, 80)
    mel = mel.t().unsqueeze(0)

    return mel


def main():
    args = get_args()

    encoder_path = args.model_type + "-encoder.onnx";
    decoder_path = args.model_type + "-decoder.onnx";
    tokens_path = args.model_type + "-tokens.txt";
    
    encoder_device = decoder_device = None;
    if args.device is not None:
        encoder_device = decoder_device = args.device.upper()

    print(f"Whisper encoder model: {encoder_path}")
    print(f"Whisper encoder device: {encoder_device}")
    print(f"Whisper decoder model: {decoder_path}")
    print(f"Whisper decoder device: {decoder_device}")
    print(f"Whisper tokens: {tokens_path}")

    model = OnnxModel(
        encoder_path, 
        decoder_path, 
        encoder_device, 
        decoder_device
    )
    n_mels = model.n_mels
    n_text_ctx = model.n_text_ctx

    mel = compute_features(args.sound_file, dim=n_mels)

    n_layer_cross_k, n_layer_cross_v = model.run_encoder(mel)

    if args.language is not None:
        if model.is_multilingual is False and args.language != "en":
            print(f"This model supports only English. Given: {args.language}")
            return

        if args.language not in model.lang2id:
            print(f"Invalid language: {args.language}")
            print(f"Valid values are: {list(model.lang2id.keys())}")
            return

        # [sot, lang, task, notimestamps]
        model.sot_sequence[1] = model.lang2id[args.language]
    elif model.is_multilingual is True:
        print("detecting language")
        lang = model.detect_language(n_layer_cross_k, n_layer_cross_v)
        model.sot_sequence[1] = lang

    if args.task is not None:
        if model.is_multilingual is False and args.task != "transcribe":
            print("This model supports only English. Please use --task=transcribe")
            return
        assert args.task in ["transcribe", "translate"], args.task

        if args.task == "translate":
            model.sot_sequence[2] = model.translate

    n_layer_self_k_cache, n_layer_self_v_cache = model.get_self_cache()
    
    offset = torch.zeros(1, dtype=torch.int64)

    print(model.sot_sequence)
 
    max_token_id = -1
    for element in model.sot_sequence:
        tokens = torch.tensor([[element]], dtype=torch.int64)
        
        attention_mask = create_attention_mask(offset.item(), max_ctx=n_text_ctx)

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
            attention_mask=attention_mask,
        )
        
        offset += 1
        
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=True)
        max_token_id = logits.argmax(dim=-1).item()

    results = []
    for i in range(model.n_text_ctx): 
        if max_token_id == model.eot:
            break
        results.append(max_token_id)
        
        tokens = torch.tensor([[results[-1]]], dtype=torch.int64)
        
        attention_mask = create_attention_mask(offset.item(), max_ctx=model.n_text_ctx)

        logits, n_layer_self_k_cache, n_layer_self_v_cache = model.run_decoder(
            tokens=tokens,
            n_layer_self_k_cache=n_layer_self_k_cache,
            n_layer_self_v_cache=n_layer_self_v_cache,
            n_layer_cross_k=n_layer_cross_k,
            n_layer_cross_v=n_layer_cross_v,
            offset=offset,
            attention_mask=attention_mask,
        )
        
        offset += 1
        
        logits = logits[0, -1]
        model.suppress_tokens(logits, is_initial=False)
        max_token_id = logits.argmax(dim=-1).item()

    token_table = load_tokens(tokens_path)
    s = b""
    for i in results:
        if i in token_table:
            s += base64.b64decode(token_table[i])

    print(f"\nTranscribed:\n{s.decode().strip()}")


if __name__ == "__main__":
    main()