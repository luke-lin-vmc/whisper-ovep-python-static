# About whisper-ovep-python-static
This Python pipeline is to show how to run Whisper on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is derived from sherpa-onnx project
https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/whisper

### Key features
* Use K-V cache to speed up inference
* Models are converted to static (mainly for NPU)


# Quick Steps
## Prepare models
Run the following commands to export models
```
pip install -r requirements.txt
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
Usage
```
Usage: python whisper_onnx.py --model_type <model_type> --device <device> voice.wav
```
* Supported model types: ```tiny``` ```base``` ```small``` ```medium``` ```large-v1``` ```large-v2``` ```large```(aka large v3) and ```turbo```(aka large v3 turbo)<br>
* Supported devices: ```CPU``` ```GPU``` ```NPU```. If ```--device``` is not specified, CPUExecutionProvider will be used by default<br> 

Run on CPU
```
python whisper_onnx.py --model_type base --device CPU how_are_you_doing_today.wav
```
Run on GPU
```
python whisper_onnx.py --model_type base --device GPU how_are_you_doing_today.wav
```
Run on NPU
```
python whisper_onnx.py --model_type base --device NPU how_are_you_doing_today.wav
```
:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation has been skipped.
## Tested Models and Devices
The test was done on a ```Intel(R) Core(TM) Ultra 7 268V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 140V GPU, driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4404 (11/7/2025)```
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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*&nbsp;Pileline worked fine but the EN speech was misdetected as PL<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**&nbsp;Pileline didn't worked due to insufficient memory

### Sample log (device is NPU)
```
(openvino_venv) C:\Github\whisper-ovep-python-static>python whisper_onnx.py --model_type base --device NPU how_are_you_doing_today.wav
Whisper encoder model: base-encoder.onnx
Whisper encoder device: NPU
Whisper decoder model: base-decoder.onnx
Whisper decoder device: NPU
Whisper tokens: base-tokens.txt
Encoder device: OpenVINO EP with device = NPU
Decoder device: OpenVINO EP with device = NPU
Encoder processing time: 62.23 ms
detecting language
Decoder processing time: 20.30 ms
detected language:  en
[50258, 50259, 50359, 50363]
Decoder processing time: 10.84 ms
Decoder processing time: 11.43 ms
Decoder processing time: 13.22 ms
Decoder processing time: 9.54 ms
Decoder processing time: 9.77 ms
Decoder processing time: 9.63 ms
Decoder processing time: 9.06 ms
Decoder processing time: 9.55 ms
Decoder processing time: 9.18 ms
Decoder processing time: 9.28 ms

Transcribed:
How are you doing today?

```
[Full log](https://github.com/luke-lin-vmc/whisper-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Known issues
1. The following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
Solution is to simply reinstall ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino
```
2. Only Arc iGPUs (Meteor Lake, Lunar Lake, Panther Lake and Arrow Lake H-series) are supported. Running on unsupported iGPU (such like Iris Xe or UHD) may lead to incorrect output, such as "!!!!!!!!!!!!!!".

