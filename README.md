# whisper-ovep-python-static plug-in (ABI) mode test
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
## Prepare OVEP plug-in DLLs
* Input ```pip show pip``` to find your Python site-packages location.
* Copy ```onnxruntime_providers_openvino_plugin.dll``` and ```onnxruntime_providers_openvino_plugin_impl.dll``` (under ```.\plugin```) into Python ```site-packages\openvino\libs```

P.S. plugin DLLs are downloaded from [Intel private repo](https://gfx-assets-build.fm.intel.com/artifactory/onnxruntime-builds/ci/develop/onnxruntime-ci-develop-236/artifacts/Windows/bdba/)

## Run
```
python whisper_onnx_ABI.py --model_type base --device NPU --mode plugin --plugin C:\Python\python313_venv\Lib\site-packages\openvino\libs\onnxruntime_providers_openvino_plugin.dll how_are_you_doing_today.wav
```
## Tested Models and Devices
The test was done on a ```Intel(R) Core(TM) Ultra 5 238V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 130V GPU (16GB), driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4621 (2/25/2026)```
### Result
| Model                     | CPU    | GPU    | NPU    |
|---------------------------|--------|--------|--------|
| base                      | OK     | OK     | OK     |
| turbo<br>(large v3 turbo) | OK     | OK     | OK     |

### Sample log (device is NPU)
```
(python313_venv) C:\GitHub\whisper-ovep-python-static>python whisper_onnx_ABI.py --model_type base --device NPU --mode plugin --plugin C:\Python\python313_venv\Lib\site-packages\openvino\libs\onnxruntime_providers_openvino_plugin.dll how_are_you_doing_today.wav
Registering OpenVINO plugin: C:\Python\python313_venv\Lib\site-packages\openvino\libs\onnxruntime_providers_openvino_plugin.dll
Whisper encoder model: base-encoder.onnx
Whisper encoder device: NPU
Whisper decoder model: base-decoder.onnx
Whisper decoder device: NPU
Whisper tokens: base-tokens.txt
Execution Mode: Plugin
Encoder device: OpenVINO EP with device = NPU (plugin mode)
Decoder device: OpenVINO EP with device = NPU (plugin mode)
Encoder processing time: 61.96 ms
detecting language
Decoder processing time: 26.31 ms
detected language:  en
[50258, 50259, 50359, 50363]
Decoder processing time: 9.85 ms
Decoder processing time: 12.01 ms
Decoder processing time: 9.58 ms
Decoder processing time: 9.69 ms
Decoder processing time: 9.74 ms
Decoder processing time: 9.01 ms
Decoder processing time: 9.72 ms
Decoder processing time: 10.26 ms
Decoder processing time: 8.85 ms
Decoder processing time: 8.80 ms

Transcribed:
How are you doing today?

Successfully unregistered Plugin EP

(python313_venv) C:\GitHub\whisper-ovep-python-static>
```
[Full log](https://github.com/luke-lin-vmc/whisper-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Reference
https://github.com/intel-innersource/frameworks.ai.onnxruntime.samples/tree/main