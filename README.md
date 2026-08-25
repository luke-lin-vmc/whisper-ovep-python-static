# About whisper-ovep-python-static
This Python pipeline shows how to run Whisper on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from sherpa-onnx project
https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/whisper

The audio sample ```how_are_you_doing_today.wav``` is downloaded from
https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

Other audio samples ("```en.wav```", "```ja.wav```" and "```zh.wav```") are downloaded from [Hugging Face sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10/tree/main/test_wavs)

### Key features
* Use plug-in mode (onnxruntime-ep-openvino)
* Use K-V cache to speed up inference
* Models are converted to static (required for NPU)

# Quick Steps
## Prepare models
Run the following commands to export models
```
pip install -r requirements.txt
```
```
python export-onnx.py --model base
```
* Supported models: ```tiny``` ```base``` ```small``` ```medium``` ```large-v1``` ```large-v2``` ```large```(aka large v3) and ```turbo```(aka large v3 turbo)<br>

Models (```*.onnx``` and ```*.weights``` (if model is large or turbo)) and tokenizer (```*tokens.txt```) will be exported under the same directory
```
base-encoder.onnx
base-decoder.onnx
base-tokens.txt
```

## Run
```
python whisper_onnx.py --model_type base --device GPU how_are_you_doing_today.wav
```
```
options:
  -h, --help            show this help message and exit
  --model_type {tiny,base,small,medium,large-v1,large-v2,large,turbo}
                        Model type
  --language LANGUAGE   The actual spoken language in the audio. Example values, en, de, zh, jp, fr. If None, language
                        will be detected automatically
  --task {transcribe,translate}
                        Valid values are: transcribe, translate. Default task is transcribe
  --device {CPU,GPU,NPU,AUTO}
                        Execution device. Use 'CPU', 'GPU', 'NPU' or 'AUTO' for OpenVINO. If not specified,
                        CPUExecutionProvider will be used by default
```
Run on CPU
```
python whisper_onnx.py --model_type base --device CPU how_are_you_doing_today.wav
```
Run on CPU, translate
```
python whisper_onnx.py --model_type base --device CPU --task translate zh.wav
```
Run on GPU
```
python whisper_onnx.py --model_type base --device GPU how_are_you_doing_today.wav
```
Run on NPU
```
python whisper_onnx.py --model_type base --device NPU how_are_you_doing_today.wav
```
Run on a AUTO selected device, the selection priority is NPU, GPU then CPU
```
python whisper_onnx.py --model_type base --device AUTO how_are_you_doing_today.wav
```
:warning:[NOTE] The 1st time running on NPU will take a long time (about 3 minutes) for model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation is skipped.
## Tested Models and Devices
The test was done on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 130V GPU (16GB), driver 32.0.101.8860 (6/25/2026)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4841 (7/24/2026)```
### Result
| Model                     | CPU    | GPU    | NPU    |
|---------------------------|--------|--------|--------|
| tiny                      | OK     | OK     | OK     |
| base                      | OK     | OK     | OK     |
| small                     | OK     | OK     | OK     |
| medium                    | OK     | OK     | OK     |
| large-v1                  | OK*    | OK*    | Fail** |
| large-v2                  | OK     | OK     | Fail** |
| large<br>(large v3)       | OK     | OK     | Fail** |
| turbo<br>(large v3 turbo) | OK     | OK     | OK     |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*&nbsp;Pipeline worked fine but the EN speech was misdetected as PL, need to specify "```--language en```" to get correct result<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**&nbsp;Pipeline didn't work due to insufficient memory

### Sample log
```
(python313_venv) C:\GitHub\whisper-ovep-python-static>python whisper_onnx.py --model_type base --device GPU how_are_you_doing_today.wav

OpenVINO Execution Provider plugin library path:
C:\Python\python313_venv\Lib\site-packages\onnxruntime_ep_openvino\onnxruntime_providers_openvino_plugin.dll

Available Execution Provider devices:
CPUExecutionProvider
OpenVINOExecutionProvider NPU
OpenVINOExecutionProvider GPU
OpenVINOExecutionProvider CPU
OpenVINOExecutionProvider.AUTO NPU
OpenVINOExecutionProvider.AUTO GPU
OpenVINOExecutionProvider.AUTO CPU

Whisper encoder model: base-encoder.onnx
Whisper decoder model: base-decoder.onnx
Whisper tokens: base-tokens.txt
Selected Execution Provider device:
OpenVINOExecutionProvider GPU

Encoder processing time: 22.81 ms
detecting language
Decoder processing time: 21.29 ms
detected language:  en
[50258, 50259, 50359, 50363]
Decoder processing time: 9.86 ms
Decoder processing time: 9.88 ms
Decoder processing time: 10.39 ms
Decoder processing time: 9.12 ms
Decoder processing time: 10.68 ms
Decoder processing time: 10.53 ms
Decoder processing time: 9.08 ms
Decoder processing time: 10.11 ms
Decoder processing time: 9.13 ms
Decoder processing time: 9.23 ms

Transcribed:
How are you doing today?
```
[Full log](https://github.com/luke-lin-vmc/whisper-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Reference
https://pypi.org/project/onnxruntime-ep-openvino/
