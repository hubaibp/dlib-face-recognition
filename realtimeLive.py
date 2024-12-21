import cv2
import os
import dlib
import numpy as np

# Initialize Dlib's face detector and face recognizer
detector = dlib.get_frontal_face_detector()
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to encode face from an image
def encode_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 1:
        shape = shape_predictor(gray, faces[0])
        return np.array(recognizer.compute_face_descriptor(img, shape))
    return None

# Train recognizer with saved faces from a folder
def train_recognizer(faces_dir):
    encodings = []
    names = []
    for file in os.listdir(faces_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(faces_dir, file)
            encoding = encode_face(path)
            if encoding is not None:
                encodings.append(encoding)
                folder_name = os.path.basename(faces_dir)  # Get the folder name (student name)
                names.append(folder_name)  # Store the folder name as the name
    return encodings, names

# Capture and save new image with name in a folder
def capture_and_save_face(folder_name):
    faces_dir = os.path.join("faces", folder_name)
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    cap = cv2.VideoCapture(0)

    print(f"Capturing images for {folder_name}... Press 'c' to capture an image and 'q' to quit.")
    
    # Capture images for this person
    image_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f"Capturing for {folder_name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture an image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) == 1:  # Only save if one face is detected
                face = faces[0]
                face_img = frame[face.top():face.bottom(), face.left():face.right()]
                filename = os.path.join(faces_dir, f"{folder_name}_{image_count + 1}.jpg")
                cv2.imwrite(filename, face_img)
                print(f"Image {image_count + 1} saved for {folder_name}")
                image_count += 1

        if key == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Recognize face in real-time
def recognize_face_in_realtime():
    cap = cv2.VideoCapture(0)

    # Load all saved students
    all_encodings = []
    all_names = []
    faces_dir = "faces"

    # Load all saved students
    for folder in os.listdir(faces_dir):
        student_dir = os.path.join(faces_dir, folder)
        if os.path.isdir(student_dir):
            encodings, names = train_recognizer(student_dir)
            all_encodings.extend(encodings)
            all_names.extend(names)

    print("Recognition started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            shape = shape_predictor(gray, face)

            # Get face encoding
            face_encoding = np.array(recognizer.compute_face_descriptor(frame, shape))

            # Compare with known encodings
            matches = [np.linalg.norm(face_encoding - enc) for enc in all_encodings]

            if matches and min(matches) < 0.4:
                # Find the closest match
                matched_idx = matches.index(min(matches))
                name = all_names[matched_idx]  # Name from the folder
                cv2.putText(frame, f"Hello {name}!", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                name = "New person"
                cv2.putText(frame, f"Hello {name}!", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to choose between capturing and recognizing
def main():
    while True:
        print("\nWelcome! Please choose one of the options below:")
        print("N - Add new student(s)")
        print("R - Recognize existing students")
        print("Q - Quit")
        print("C - Capture a new image")

        action = input("Enter your choice (N/R/Q/C): ").strip().upper()

        if action == 'N':
            # Capture images for new students
            while True:
                student_name = input("Enter student name: ")
                capture_and_save_face(student_name)

                # Ask if there are more students
                more_students = input("Do you have more students to add? (yes/no): ").strip().lower()
                if more_students != "yes":
                    print("Thanks!")
                    break
        elif action == 'R':
            recognize_face_in_realtime()
        elif action == 'C':
            # Directly capture a single image for recognition
            student_name = input("Enter student name for this capture: ")
            capture_and_save_face(student_name)
        elif action == 'Q':
            break
        else:
            print("Invalid option. Please choose 'N', 'R', 'Q', or 'C'.")

# Run the main function
if __name__ == "__main__":
    main()
