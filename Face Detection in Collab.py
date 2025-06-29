# Step 1: Install required packages
!pip install -q insightface onnxruntime
# Step 2: Import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from insightface.app import FaceAnalysis
from google.colab import files
# Step 3: Upload your image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]  # Use the first uploaded file
# Step 4: Load image
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image loading failed!")
# Step 5: Initialize the face detector
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' if you have GPU
app.prepare(ctx_id=0, det_size=(640, 640))
# Step 6: Detect faces
faces = app.get(img)
# Step 7: TEMP FIX for np.int error
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
# Now draw the detections
img_with_faces = app.draw_on(img, faces)
# Step 8: Save and display result
output_filename = "face_detected_output.jpg"
cv2.imwrite(output_filename, img_with_faces)
# Display
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Faces using SCRFD')
plt.show()
#Download the result
files.download(output_filename)

# Below one is for faces detection in a video

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from google.colab.patches import cv2_imshow
# Fix for deprecated np.int (for compatibility with InsightFace)
if not hasattr(np, 'int'):
    np.int = int
# Initialize FaceAnalysis
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
# Input Video
input_path = "/content/drive/MyDrive/Colab Notebooks/faceImages/Video1"  # Replace with your file path
cap = cv2.VideoCapture(input_path)
# Output Video Writer Setup
output_path = "/content/drive/MyDrive/Colab Notebooks/faceImages/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# Process Frame-by-Frame
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = app.get(frame)
    frame = app.draw_on(frame, faces)
    out.write(frame)             # Save frame to output video
    if frame_count % 10 == 0:    # Show every 10th frame in Colab (optional)
        print(f"Processed frame {frame_count}")
        cv2_imshow(frame)
    frame_count += 1
# Cleanup
cap.release()
out.release()
print(f"Face detection complete. Saved to: {output_path}")
