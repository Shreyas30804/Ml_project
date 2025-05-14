import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image

# Initialize FastAPI
app = FastAPI()


# Load Models (relative paths for deployment)
face_model = cv2.dnn.readNet(
    "models/opencv_face_detector_uint8.pb",
    "models/opencv_face_detector.pbtxt"
)

gender_model = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)

age_model = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)


# Class Labels
age_classes = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classes = ['Male', 'Female']

# Mean values for preprocessing
mean_values = [104, 117, 123]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        image = np.array(image)

        if image is None:
            return {"error": "Unable to load image."}

        # Convert to OpenCV format and resize
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), mean_values, swapRB=True, crop=False)

        # Detect faces
        face_model.setInput(blob)
        detections = face_model.forward()

        face_count = 0
        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # Adjust confidence threshold as needed
                face_count += 1

                # Get face coordinates
                h, w = image.shape[:2]
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                # Extract face region
                face = image[max(0, y1):min(y2, h - 1), max(0, x1):min(x2, w - 1)]
                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean_values, swapRB=True)

                # Gender Prediction
                gender_model.setInput(face_blob)
                gender_preds = gender_model.forward()
                gender = gender_classes[np.argmax(gender_preds)]

                # Age Prediction
                age_model.setInput(face_blob)
                age_preds = age_model.forward()
                age = age_classes[np.argmax(age_preds)]

                # Append results
                results.append({"gender": gender, "age": age, "confidence": float(confidence)})

        if face_count == 0:
            return {"error": "No faces detected in the image."}

        return {"face_count": face_count, "results": results}

    except Exception as e:
        return {"error": str(e)}
 