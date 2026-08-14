# whisper-ovep-python-static plug-in mode (onnxruntime-ep-openvino)
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
```
python whisper_onnx.py --model_type base --device GPU how_are_you_doing_today.wav
```
## Tested Models and Devices
The test was done on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 130V GPU (16GB), driver 32.0.101.8860 (6/25/2026)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4841 (7/24/2026)```
### Result
| Model                     | CPU    | GPU    | NPU    |
|---------------------------|--------|--------|--------|
| base                      | OK     | OK     | NG     |
| small                     | OK     | OK     | NG     |
| turbo<br>(large v3 turbo) | OK     | OK     | NG     |

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

Encoder processing time: 31.10 ms
detecting language
Decoder processing time: 35.12 ms
detected language:  en
[50258, 50259, 50359, 50363]
Decoder processing time: 12.44 ms
Decoder processing time: 16.70 ms
Decoder processing time: 13.47 ms
Decoder processing time: 13.05 ms
Decoder processing time: 13.36 ms
Decoder processing time: 13.28 ms
Decoder processing time: 13.08 ms
Decoder processing time: 12.50 ms
Decoder processing time: 12.58 ms
Decoder processing time: 13.42 ms

Transcribed:
How are you doing today?
```
## Reference
https://pypi.org/project/onnxruntime-ep-openvino/