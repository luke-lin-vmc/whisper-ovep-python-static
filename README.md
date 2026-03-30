# whisper-ovep-python-static plug-in (ABI) mode
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
python whisper_onnx.py --model_type base --device NPU --plugin C:\Python\python313_venv\Lib\site-packages\openvino\libs\onnxruntime_providers_openvino_plugin.dll how_are_you_doing_today.wav
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
(python313_venv) C:\GitHub\whisper-ovep-python-static>python whisper_onnx.py --model_type base --device NPU --plugin C:\Python\python313_venv\Lib\site-packages\openvino\libs\onnxruntime_providers_openvino_plugin.dll how_are_you_doing_today.wav
Registering execution provider: OpenVINOExecutionProvider, plugin: C:\Python\python313_venv\Lib\site-packages\openvino\libs\onnxruntime_providers_openvino_plugin.dll
Whisper encoder model: base-encoder.onnx
Whisper decoder model: base-decoder.onnx
Whisper tokens: base-tokens.txt
Inference device: NPU
@@@@@ ep_device.ep_metadata = {'ov_device': 'NPU', 'version': '1.2.0-dev+99f5532d5'}
Encoder processing time: 59.59 ms
detecting language
Decoder processing time: 23.05 ms
detected language:  en
[50258, 50259, 50359, 50363]
Decoder processing time: 9.71 ms
Decoder processing time: 11.79 ms
Decoder processing time: 9.87 ms
Decoder processing time: 9.32 ms
Decoder processing time: 9.34 ms
Decoder processing time: 10.46 ms
Decoder processing time: 14.51 ms
Decoder processing time: 11.45 ms
Decoder processing time: 10.26 ms
Decoder processing time: 9.46 ms

Transcribed:
How are you doing today?

Successfully unregistered execution provider: OpenVINOExecutionProvider

(python313_venv) C:\GitHub\whisper-ovep-python-static>
```
## Reference
https://github.com/intel-innersource/frameworks.ai.onnxruntime.samples/tree/main