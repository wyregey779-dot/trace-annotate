# YuNet face detector model card (local copy)

- File: `face_detection_yunet_2023mar.onnx`
- SHA256: `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4`
- Size: 232589 bytes
- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Download URL used: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- License: MIT (model card and redistributions list MIT; commercial use allowed)
- Runtime: OpenCV `cv2.FaceDetectorYN`, fully local; no image or video leaves the machine.
- Role in TraceAnnotate: face boxes only. It never produces an identity, emotion,
  or shot-size label. Shot size is a separate geometry rule and is downgraded
  when only a face box is available.

License evidence checked on 2026-11-06:
- https://models.luxonis.com/luxonis/yunet/5d635f3c-45c0-41d2-8800-7ca3681b1915
- https://www.kaggle.com/models/rajarshi2712/yunet-face-detect/other/default/1
