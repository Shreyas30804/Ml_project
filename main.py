import cv2
import numpy as np
import os

# Image Path
image_path = r"C:/BACKUP NEVER DELETE/Desktop/Project/man2.jpg"

# Ensure Image Exists
if not os.path.exists(image_path):
    print("Error: Image file not found. Please check the path.")
    exit()

# Load Image
image = cv2.imread(image_path)
if image is None:
    print("Error: Failed to load the image. Please check the file format.")
    exit()

# image = cv2.resize(image, (720, 640))
screen_width, screen_height = image.shape[1], image.shape[0]
scaling_factor = 0.5  # Adjust based on your phone's performance
image = cv2.resize(image, (int(screen_width * scaling_factor), int(screen_height * scaling_factor)))

img_cp = image.copy()

# Model Paths
face_pbtxt = r"C:/BACKUP NEVER DELETE/Desktop/Project/models/opencv_face_detector.pbtxt"
face_pb = r"C:/BACKUP NEVER DELETE/Desktop/Project/models/opencv_face_detector_uint8.pb"
age_prototxt = r"C:/BACKUP NEVER DELETE/Desktop/Project/models/age_deploy.prototxt"
age_caffe = r"C:/BACKUP NEVER DELETE/Desktop/Project/models/age_net.caffemodel"
gender_prototxt = r"C:/BACKUP NEVER DELETE/Desktop/Project/models/gender_deploy.prototxt"
gender_caffemodel = r"C:/BACKUP NEVER DELETE/Desktop/Project/models/gender_net.caffemodel"

# Mean Values for Normalization
mean_values = [104, 117, 123]

# Load Models
face = cv2.dnn.readNet(face_pb, face_pbtxt)
gen = cv2.dnn.readNet(gender_caffemodel, gender_prototxt)
age = cv2.dnn.readNet(age_caffe, age_prototxt)

# Classifications
age_classification = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classification = ['Male', 'Female']

# Get Image Dimensions
img_h, img_w = img_cp.shape[:2]

# Blob Creation for Face Detection
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), mean_values, swapRB=True, crop=False)
face.setInput(blob)
detected_faces = face.forward()

# Face Detection
face_bounds = []
for i in range(detected_faces.shape[2]):
    confidence = detected_faces[0, 0, i, 2]
    if confidence > 0.5:
        x1 = int(detected_faces[0, 0, i, 3] * img_w)
        y1 = int(detected_faces[0, 0, i, 4] * img_h)
        x2 = int(detected_faces[0, 0, i, 5] * img_w)
        y2 = int(detected_faces[0, 0, i, 6] * img_h)
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_bounds.append([x1, y1, x2, y2])

# Gender and Age Prediction
for face_bound in face_bounds:
    try:
        x1, y1, x2, y2 = face_bound
        face_img = img_cp[max(0, y1 - 15):min(y2 + 15, img_cp.shape[0] - 1),
                          max(0, x1 - 15):min(x2 + 15, img_cp.shape[1] - 1)]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, img_w - 1), min(y2, img_h - 1)


        # Blob for Age and Gender Detection
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), mean_values, swapRB=True)
        
        # Gender Prediction
        gen.setInput(blob)
        gender_prediction = gen.forward()
        gender_index = np.argmax(gender_prediction[0])
        gender = gender_classification[gender_index]

        # Age Prediction
        age.setInput(blob)
        age_prediction = age.forward()
        age_index = np.argmax(age_prediction[0])
        age_range = age_classification[age_index]

        # Display Result
        text = f"{gender}, {age_range}"
        cv2.putText(img_cp, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(text)
    
    except Exception as e:
        print(f"Error in face region: {e}")
        continue

# Display Result
cv2.imshow('Result', img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()
