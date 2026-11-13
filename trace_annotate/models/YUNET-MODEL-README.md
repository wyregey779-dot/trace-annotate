# YuNet face detector model card (local copy)

- File: `face_detection_yunet_2023mar.onnx`
- SHA256: `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4`
- Size: 232589 bytes
- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Download URL used: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- License: [MIT, original copyright and full notice](YUNET-LICENSE.txt).
- Runtime: OpenCV `cv2.FaceDetectorYN`, fully local; no image or video leaves the machine.
- Role in TraceAnnotate: face boxes only. It never produces an identity, emotion,
  or shot-size label. Shot size is a separate geometry rule and is downgraded
  when only a face box is available.

License evidence checked on 2026-11-06:
- https://models.luxonis.com/luxonis/yunet/5d635f3c-45c0-41d2-8800-7ca3681b1915
- https://www.kaggle.com/models/rajarshi2712/yunet-face-detect/other/default/1

Official upstream provenance checked on 2026-11-13:
- Revision: `48b10fa505aeb51fbd127fc60a5c660040e823b7` in OpenCV Zoo.
- [Pinned upstream model directory](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet).
- [Pinned model LFS pointer](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) records the SHA256 and size above.
- [Original MIT notice](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE) has SHA256 `c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958`.
- Redistribution must retain Shiqi Yu's copyright and the full MIT notice.
  `YUNET-LICENSE.txt` is included with the model and in distribution license metadata.

The detector initializes at 320x320. Each inference sets its input size to the
actual decoded frame width and height. New model-run records describe this
per-frame sizing policy. Historical records retain their original metadata.
