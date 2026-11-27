# OpenVINO Driver Action Recognition ADAS 0002

Pretrained Intel Open Model Zoo composite model for in-cabin driver action
recognition. The checked-in runtime files use the FP16-INT8 precision variant.

## Runtime files

- `FP16-INT8/driver-action-recognition-adas-0002-encoder.xml`
- `FP16-INT8/driver-action-recognition-adas-0002-encoder.bin`
- `FP16-INT8/driver-action-recognition-adas-0002-decoder.xml`
- `FP16-INT8/driver-action-recognition-adas-0002-decoder.bin`

## Input contract

1. Sample 16 BGR frames over approximately one second.
2. Resize each frame to 224 x 224.
3. Run the encoder on each frame using NCHW input `[1, 3, 224, 224]`.
4. Stack the 16 encoder embeddings into decoder input `[1, 16, 512]`.
5. Apply softmax to the decoder output `[1, 9]`.

## Class order

| ID | Open Model Zoo label | MDDAD mapping |
|---:|---|---|
| 0 | Safe driving | `normal_driving` |
| 1 | Texting left | `phone_texting` |
| 2 | Texting right | `phone_texting` |
| 3 | Talking phone left | `phone_calling` |
| 4 | Talking phone right | `phone_calling` |
| 5 | Operating radio | `other_distraction` |
| 6 | Drinking eating | `other_distraction` |
| 7 | Reaching behind | `other_distraction` |
| 8 | Hair and makeup | `other_distraction` |

The model makes the behavior decision from a video clip. Existing RT-DETR phone
detectors may be used as supporting visual evidence, but their object detections
must not override the action model prediction.

## Provenance

- Model: `driver-action-recognition-adas-0002`
- Source: https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/driver-action-recognition-adas-0002
- License: Apache-2.0
