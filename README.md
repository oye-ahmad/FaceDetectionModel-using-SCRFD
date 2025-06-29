# Face Detection using InsightFace SCRFD

This project demonstrates real-time face detection on images and video using the SCRFD model from [InsightFace](https://github.com/deepinsight/insightface). It uses ONNXRuntime for efficient inference and OpenCV for image and video handling. The system can annotate faces in static images or process video files frame by frame to highlight detected faces.

---

## 🚀 Features

- Lightweight, fast, and accurate SCRFD face detection
- Supports static images and video input
- Annotates bounding boxes and facial landmarks
- Runs on CPU or GPU (if available)
- Modular and easy to extend (e.g. for webcam input)

---

## 🛠️ Libraries and Tools Used

- **OpenCV (cv2)**: Image and video reading, writing, and annotation
- **NumPy**: Numerical operations and array manipulations
- **Matplotlib**: Visualization for annotated images
- **InsightFace**: Deep learning-based face analysis (detection, recognition, alignment)
- **ONNXRuntime**: Executes ONNX models efficiently

---

## Output

![facesDetected](https://github.com/user-attachments/assets/5e9f8e52-2b24-496e-9551-fdf608178521)


## 📦 Installation

You can install the required libraries using pip:

pip install insightface onnxruntime opencv-python matplotlib

If you're using Google Colab, you can install directly:

!pip install -q insightface onnxruntime

---

## 🛠️ How it Works

**Loads Pretrained SCRFD ONNX Model:** Fast, lightweight detector trained for face localization.

**Processes Input:** Image or video frame.

**Runs Detection:** Identifies faces and landmarks.

**Annotates:** Draws bounding boxes and keypoints.

**Outputs:** Saves processed image or video.

## ✅ Conclusion

This project demonstrates real-time face detection using the InsightFace SCRFD model with ONNXRuntime and OpenCV. It supports images and videos, is easy to customize, and offers high performance on CPU or GPU.

Feel free to fork the repo and adapt it for webcam streams, batch processing, or integration into larger systems!

