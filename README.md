# nuzky-models

ONNX conversions of three Apache-2.0 MediaPipe models, for [Nuzky](https://nuzky.app), the video
editor, which picks cover frames and finds people with them. Nuzky downloads these files from this
repository at a pinned commit and checks their size and SHA-256 before using them.

| File | Bytes | SHA-256 | Source |
|---|---|---|---|
| `face-landmarks-v2.onnx` | 4831591 | `6fc4bae7e3e2c5c0870e8d1b764839c7dcefee7e96c874b454da8be514e19f47` | `face_landmarks_detector.tflite` in [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) (SHA-256 `64184e22…6544e0bc9ff`) |
| `face-blendshapes-v2.onnx` | 1822957 | `74029fcef4076695dd1129d9c45a3d766532745868bd1eb3cc1deff87e7d6d66` | `face_blendshapes.tflite` in the same task file |
| `selfie-segmenter.onnx` | 445668 | `7759b1df460279b03bb39813ee1cec7bc3a1a4be38006e98ddd48294c92bd9f4` | [selfie_segmenter.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite) (SHA-256 `191ac952…e4038226658b`) |

## Licence

Apache License 2.0, as the originals. The model cards say so:
[Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf),
[Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf),
[Selfie Segmentation](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf).
What was changed is in `NOTICE`.

## How they were made

`convert-mediapipe.py` downloads the pinned sources, checks their SHA-256, converts them with
tf2onnx (opset 17, float32 weights) and runs the originals and ONNX Runtime 1.28 on the same
inputs: 48 real face crops from public-domain NASA and NOAA interviews plus perturbed and
synthetic inputs. It fails when they differ by more than 0.05 px (landmarks) or 1e-3 (scores
and mask). Measured maximum differences: landmarks 3.4e-4 px, blendshapes 2.4e-6, person mask
8.2e-5. The selfie segmenter's custom transposed convolution is rebuilt as a standard
`ConvTranspose`; its reference is MediaPipe's own ImageSegmenter, since plain LiteRT cannot load
the custom op. Two runs give byte-identical files.

The script here is a copy of `scripts/convert-mediapipe.py` in the Nuzky repository and runs
from a Nuzky checkout:

```
UV_CACHE_DIR=tmp-test/vision-convert/uv-cache uv run --script scripts/convert-mediapipe.py
```

## Tensors

- `face-landmarks-v2.onnx`: input `image` [1,256,256,3] RGB in 0..1, a square face crop 1.5 times
  the face box, rotated so the eyes are level. Outputs `landmarks` [1,1,1,1434] (478 x, y, z in crop
  pixels), `presence_logit` [1,1,1,1] (sigmoid, threshold 0.5), `tongue_out` [1,1].
- `face-blendshapes-v2.onnx`: input `points` [1,146,2], MediaPipe's 146-landmark subset in image
  pixels. Output `blendshapes` [52] in MediaPipe's category order.
- `selfie-segmenter.onnx`: input `image` [1,256,256,3] RGB in 0..1, the whole frame stretched.
  Output `mask` [1,256,256,1], person probability.
