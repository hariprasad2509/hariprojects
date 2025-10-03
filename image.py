import cv2
import numpy as np
import os
from pathlib import Path

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir="knownimages"):
        self.known_faces_dir = known_faces_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels_dict = {}  # label_id -> person_name
        self.is_trained = False

    def load_known_faces(self):
        faces = []
        labels = []
        label_id = 0

        subfolders = [f for f in os.listdir(self.known_faces_dir) if os.path.isdir(os.path.join(self.known_faces_dir, f))]
        if not subfolders:
            print("No subfolders found in knownimages!")
            return False

        for folder in subfolders:
            person_folder = os.path.join(self.known_faces_dir, folder)
            self.labels_dict[label_id] = folder  # map label to person name
            print(f"Processing images for: {folder}")

            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(person_folder).glob(ext))
                image_files.extend(Path(person_folder).glob(ext.upper()))

            for image_path in image_files:
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in detected_faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    faces.append(face)
                    labels.append(label_id)
                    print(f"Processed {image_path.name}")

            label_id += 1

        if len(faces) == 0:
            print("No faces detected!")
            return False

        self.face_recognizer.train(faces, np.array(labels))
        self.is_trained = True
        print(f"Training completed! Recognizes: {list(self.labels_dict.values())}")
        return True

    def recognize_face(self, face_gray):
        if not self.is_trained:
            return "Unknown", 0

        face_resized = cv2.resize(face_gray, (200, 200))
        label, confidence = self.face_recognizer.predict(face_resized)

        threshold = 70
        if confidence < threshold:
            name = self.labels_dict.get(label, "Unknown")
        else:
            name = "Unknown"
        return name, confidence

    def start_recognition(self):
        if not self.is_trained:
            print("Please load and train faces first!")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Face recognition started! Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                name, confidence = self.recognize_face(face_gray)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    fr_system = FaceRecognitionSystem("knownimages")
    if fr_system.load_known_faces():
        fr_system.start_recognition()
    else:
        print("Failed to load known faces.")

if __name__ == "__main__":
    main()
