import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# OpenCV Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Transform for input face images
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5)

def extract_face(image, box):
    x, y, w, h = box
    return image[y:y+h, x:x+w]

def get_embedding(face_image):
    face = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face).unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding[0].numpy()

def load_known_faces(directory):
    embeddings = {}
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.png')):
            name = os.path.splitext(file)[0]
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            boxes = detect_faces(img)
            if len(boxes) == 0:
                print(f"[WARNING] No face in {file}")
                continue
            face = extract_face(img, boxes[0])
            emb = get_embedding(face)
            embeddings[name] = emb
            print(f"[INFO] Loaded: {name}")
    return embeddings

def recognize_face(embedding, known_faces, threshold=0.6):
    best_match = "Unknown"
    best_score = 0
    for name, known_emb in known_faces.items():
        sim = cosine_similarity([embedding], [known_emb])[0][0]
        if sim > best_score and sim > threshold:
            best_score = sim
            best_match = name
    return best_match

def main():
    known_faces_dir = "known_faces"
    print("[INFO] Loading known faces...")
    known_faces = load_known_faces(known_faces_dir)

    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = extract_face(frame, (x, y, w, h))
            embedding = get_embedding(face_img)
            name = recognize_face(embedding, known_faces)

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
