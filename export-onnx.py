#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
# flake8: noqa

"""
Note: Code in this file is modified from
https://github.com/TadaoYamaoka/whisper/blob/main/to_onnx.py

Thanks to https://github.com/TadaoYamaoka
for making the onnx export script public.

Note that we have removed the 30 seconds constraint from whisper. You can
use any T <= 30.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

import whisper
from whisper.model import (
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        # fmt: off
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large-v1", "large-v2",
            "large", "large-v3", "turbo", # these three have feature dim 128
            "distil-medium.en", "distil-small.en", "distil-large-v2",
            "distil-large-v3",
            "distil-large-v3.5",
            # for fine-tuned models from icefall
            "medium-aishell",
            ],
        # fmt: on
    )
    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if "large" in filename or "turbo" in filename:
        external_filename = filename.split(".onnx")[0]
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )
    else:
        onnx.save(model, filename)


def modified_audio_encoder_forward(self: AudioEncoder, x: torch.Tensor):
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    assert (
        x.shape[2] == self.positional_embedding.shape[1]
    ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
    assert (
        x.shape[1] <= self.positional_embedding.shape[0]
    ), f"input audio length {x.shape[1]} exceeds max context {self.positional_embedding.shape[0]}"
    
    x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x


AudioEncoder.forward = modified_audio_encoder_forward


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x) # (N_audio, T_ctx, D)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))
        
        cross_k = torch.stack(n_layer_cross_k_list).permute(1, 0, 2, 3) 
        cross_v = torch.stack(n_layer_cross_v_list).permute(1, 0, 2, 3)

        return cross_k, cross_v


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        # Note: The underlying qkv_attention is patched globally to ensure ONNX compatibility.
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        current_index: Tensor,
        attention_mask: Tensor,
    ):
        q = self.multiHeadAttention.query(x)  # (b, 1, n_state)
        k = self.multiHeadAttention.key(x)    # (b, 1, n_state)
        v = self.multiHeadAttention.value(x)  # (b, 1, n_state)

        k_cache_updated = k_cache.clone()
        v_cache_updated = v_cache.clone()
        
        batch_size = k_cache.shape[0]
        # target_index shape: [B, 1, 1]
        target_index = current_index.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)

        # source shape: [B, 1, D]
        source_k = k 
        source_v = v

        k_cache_updated = k_cache_updated.scatter_(1, target_index.expand_as(source_k), source_k)
        v_cache_updated = v_cache_updated.scatter_(1, target_index.expand_as(source_v), source_v)

        wv, qk = self.multiHeadAttention.qkv_attention(
            q, k_cache_updated, v_cache_updated, attention_mask
        )
        
        return self.multiHeadAttention.out(wv), k_cache_updated, v_cache_updated


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = (
            MultiHeadAttentionCross(inResidualAttentionBlock.cross_attn)
            if inResidualAttentionBlock.cross_attn
            else None
        )

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        current_index: Tensor,
        attention_mask: Tensor,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.originalBlock.attn_ln(x), 
            self_k_cache,
            self_v_cache,
            current_index,
            attention_mask,
        )
        x = x + self_attn_x

        if self.cross_attn:
            x = x + self.cross_attn(
                self.originalBlock.cross_attn_ln(x), cross_k, cross_v
            )

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated


class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlockTensorCache(orginal_block))

    def forward(
        self,
        tokens: Tensor,
        n_layer_self_k_cache: Tensor,
        n_layer_self_v_cache: Tensor,
        n_layer_cross_k: Tensor,
        n_layer_cross_v: Tensor,
        current_index: Tensor,
        attention_mask: Tensor,
    ):
        index = current_index.squeeze()
        
        pos_emb = self.textDecoder.positional_embedding.select(0, index)
        
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(0).expand_as(self.textDecoder.token_embedding(tokens))

        x = self.textDecoder.token_embedding(tokens) + pos_emb
        x = x.to(n_layer_cross_k.dtype)

        assert x.shape[1] == 1, "tokens must have length 1 for single-step ONNX export."

        i = 0
        num_layers = len(self.blocks)
        
        out_n_layer_self_k_cache = n_layer_self_k_cache.clone()
        out_n_layer_self_v_cache = n_layer_self_v_cache.clone()

        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[:, i]
            self_v_cache = n_layer_self_v_cache[:, i]
            
            cross_k = n_layer_cross_k[:, i]
            cross_v = n_layer_cross_v[:, i]
            
            x, self_k_cache_updated, self_v_cache_updated = block(
                x,
                self_k_cache, 
                self_v_cache, 
                cross_k=cross_k,
                cross_v=cross_v,
                current_index=current_index,
                attention_mask=attention_mask,
            )
            
            out_n_layer_self_k_cache[:, i] = self_k_cache_updated
            out_n_layer_self_v_cache[:, i] = self_v_cache_updated
            i += 1
        
        x = self.textDecoder.ln(x)

        logits = (
            torch.matmul(
                self.textDecoder.token_embedding.weight.to(x.dtype),
                x.permute(0, 2, 1),
            )
            .permute(0, 2, 1)
            .float()
        )

        return logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache


def patched_qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    n_head = self.n_head
    d_model = q.shape[-1]
    d_head = d_model // n_head
    
    scale = (d_head ** -0.5)

    q = q.view(*q.shape[:-1], n_head, d_head).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:-1], n_head, d_head).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:-1], n_head, d_head).permute(0, 2, 1, 3)

    qk = torch.matmul(q, k.transpose(-2, -1)) * scale

    if mask is not None:
        qk = qk + mask

    w = F.softmax(qk, dim=-1).to(q.dtype)
    
    wv = torch.matmul(w, v)
    
    wv = wv.permute(0, 2, 1, 3).flatten(start_dim=2)

    return wv, qk

MultiHeadAttention.qkv_attention = patched_qkv_attention

# ref: https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py#L232
def convert_tokens(name, model):
    whisper_dir = Path(whisper.__file__).parent
    multilingual = model.is_multilingual
    tokenizer = (
        whisper_dir
        / "assets"
        / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
    )
    if not tokenizer.is_file():
        raise ValueError(f"Cannot find {tokenizer}")

    #import base64

    with open(tokenizer, "r") as f:
        contents = f.read()
        #  tokens = {
        #      base64.b64decode(token): int(rank)
        #      for token, rank in (line.split() for line in contents.splitlines() if line)
        #  }
        tokens = {
            token: int(rank)
            for token, rank in (line.split() for line in contents.splitlines() if line)
        }

    with open(f"{name}-tokens.txt", "w") as f:
        for t, i in tokens.items():
            f.write(f"{t} {i}\n")


@torch.no_grad()
def main():
    args = get_args()
    name = args.model
    
    opset_version = 17

    if name == "distil-medium.en":
        filename = "./distil-medium-en-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-medium.en
                to download original-model.bin
                You can use the following command to do that:

                wget -O distil-medium-en-original-model.bin https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/original-model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-large-v2":
        filename = "./distil-large-v2-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-large-v2
                to download original-model.bin
                You can use the following command to do that:

                wget -O distil-large-v2-original-model.bin https://huggingface.co/distil-whisper/distil-large-v2/resolve/main/original-model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-large-v3":
        filename = "./distil-large-v3-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-large-v3-openai
                to download model.bin
                You can use the following command to do that:

                wget -O distil-large-v3-original-model.bin https://huggingface.co/distil-whisper/distil-large-v3-openai/resolve/main/model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-large-v3.5":
        filename = "./distil-large-v3.5-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-large-v3.5-openai/
                to download model.bin
                You can use the following command to do that:

                wget -O distil-large-v3.5-original-model.bin https://huggingface.co/distil-whisper/distil-large-v3.5-openai/resolve/main/model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "distil-small.en":
        filename = "./distil-small-en-original-model.bin"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/distil-whisper/distil-small.en
                to download original-model.bin
                You can use the following command to do that:

                wget -O distil-small-en-original-model.bin https://huggingface.co/distil-whisper/distil-small.en/resolve/main/original-model.bin
            """
            )
        model = whisper.load_model(filename)
    elif name == "medium-aishell":
        filename = "./medium-aishell.pt"
        if not Path(filename).is_file():
            raise ValueError(
                """
                Please go to https://huggingface.co/yuekai/icefall_asr_aishell_whisper/tree/main/exp_medium
                to download whisper-medium-aishell1-epoch-10-avg-4.pt
                You can use the following command to do that:

                wget -O medium-aishell.pt https://huggingface.co/yuekai/icefall_asr_aishell_whisper/resolve/main/exp_medium/whisper-medium-aishell1-epoch-10-avg-4.pt
            """
            )
        model = whisper.load_model(filename)
    else:
        model = whisper.load_model(name)
        
    print(model.dims)

    print(
        f"number of model parameters: {name}",
        sum(p.numel() for p in model.parameters()),
    )
    print(
        f"number of encoder parameters: {name}",
        sum(p.numel() for p in model.encoder.parameters()),
    )
    print(
        f"number of decoder parameters: {name}",
        sum(p.numel() for p in model.decoder.parameters()),
    )

    convert_tokens(name=name, model=model)

    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )

    model.eval()
    print(model.dims)
    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)
    assert audio.shape == (16000 * 30,), audio.shape

    if args.model in ("distil-large-v3", "distil-large-v3.5"):
        n_mels = 128
    elif args.model in (
        "large",
        "large-v3",
        "turbo",
    ):
        n_mels = 128
    else:
        n_mels = 80

    mel = (
        whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device).unsqueeze(0)
    )
    batch_size = 1
    assert mel.shape == (batch_size, n_mels, 30 * 100), mel.shape

    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)

    n_layer_cross_k, n_layer_cross_v = encoder(mel)
    
    encoder_filename = f"{name}-encoder.onnx"
    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
    )

    encoder_meta_data = {
        "model_type": f"whisper-{name}",
        "version": "1",
        "maintainer": "k2-fsa",
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
        "sot_sequence": ",".join(list(map(str, tokenizer.sot_sequence))),
        "all_language_tokens": ",".join(
            list(map(str, tokenizer.all_language_tokens))
        ),  # a list of ids
        "all_language_codes": ",".join(
            tokenizer.all_language_codes
        ),  # e.g., en, de, zh, fr
        "sot": tokenizer.sot,
        "sot_index": tokenizer.sot_sequence.index(tokenizer.sot),
        "eot": tokenizer.eot,
        "blank_id": tokenizer.encode(" ")[0],
        "is_multilingual": int(model.is_multilingual),
        "no_speech": tokenizer.no_speech,
        "non_speech_tokens": ",".join(list(map(str, tokenizer.non_speech_tokens))),
        "transcribe": tokenizer.transcribe,
        "translate": tokenizer.translate,
        "sot_prev": tokenizer.sot_prev,
        "sot_lm": tokenizer.sot_lm,
        "no_timestamps": tokenizer.no_timestamps,
    }
    print(f"encoder_meta_data: {encoder_meta_data}")
    add_meta_data(filename=encoder_filename, meta_data=encoder_meta_data)

    n_audio = mel.shape[0]
    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)
    max_text_ctx = model.dims.n_text_ctx

    n_layer_self_k_cache = torch.zeros(
        (
            n_audio, # n_audio = 1 (Batch Size)
            len(model.decoder.blocks), # N_layer = 12
            max_text_ctx, # N_ctx = 448
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            n_audio, # n_audio = 1 (Batch Size)
            len(model.decoder.blocks), # N_layer = 12
            max_text_ctx, # N_ctx = 448
            model.dims.n_text_state,
        ),
        device=mel.device,
    )

    initial_tokens = [tokenizer.sot, tokenizer.sot, tokenizer.sot]
    initial_tokens_len = len(initial_tokens) 
    
    current_index_prime = torch.zeros(1, dtype=torch.int64).to(mel.device)
    
    mask_prime = torch.zeros(
        (n_audio, 1, 1, max_text_ctx), 
        dtype=torch.float32, 
        device=mel.device
    ) 

    print(f"Start pre-filling cache with {initial_tokens_len} tokens...")

    for i, token_id in enumerate(initial_tokens):
        tokens_input = torch.tensor([[token_id]] * n_audio).to(mel.device) 
        
        current_index_prime[0] = i # i = 0, 1, 2
        

        _, n_layer_self_k_cache, n_layer_self_v_cache = decoder(
            tokens_input,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            current_index_prime, 
            mask_prime,
        )
    
    print("Pre-filling complete.")
    
    tokens = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio=1, n_tokens=1]
    
    current_index = torch.tensor([initial_tokens_len], dtype=torch.int64).to(mel.device) 

    attention_mask = torch.full(
        (n_audio, 1, 1, max_text_ctx), 
        -torch.inf,
        device=mel.device
    )
    attention_mask[:, :, :, : current_index.item() + tokens.shape[1]] = 0 
    
    logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        current_index,
        attention_mask,
    )
    
    assert logits.shape == (n_audio, tokens.shape[1], model.dims.n_vocab)
    assert out_n_layer_self_k_cache.shape == (
        n_audio, # 1
        model.dims.n_text_layer, # 12
        max_text_ctx, # 448
        model.dims.n_text_state, # 768
    )

    decoder_filename = f"{name}-decoder.onnx"
    torch.onnx.export(
        decoder,
        (
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            current_index,
            attention_mask,
        ),
        decoder_filename,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "current_index",
            "attention_mask",
        ],
        output_names=["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
    )

    if "large" in args.model:
        decoder_external_filename = decoder_filename.split(".onnx")[0]
        decoder_model = onnx.load(decoder_filename)
        onnx.save(
            decoder_model,
            decoder_filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=decoder_external_filename + ".weights",
        )

if __name__ == "__main__":
    main()
