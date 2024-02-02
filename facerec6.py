import face_recognition as fr
import numpy as np
import cv2
import os
from datetime import datetime

# Path to folder containing all known faces.
faces_path = "C:\\vs codeee 00\\imgd"
unknown = "unknown"

# Function to get face names, as well as face encodings
def add_new_face(name, face_encoding):
    # Add a new face to the database
    face_names.append(name)
    face_encodings.append(face_encoding)

# Function to mark attendance for known faces
def mark_attendance(name):
    if name not in attendance:
        print(f"{name} is present!")
        attendance.append(name)

def save_attendance_report():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Folder for attendance reports
    attendance_report_folder = os.path.join(os.getcwd(), "attendance_reports")
    os.makedirs(attendance_report_folder, exist_ok=True)

    # Folder for present reports
    present_report_folder = os.path.join(attendance_report_folder, "present")
    os.makedirs(present_report_folder, exist_ok=True)
    present_report_path = os.path.join(present_report_folder, f"attendance_report_{current_time}.txt")

    with open(present_report_path, "w") as present_report_file:
        present_report_file.write(f"Present:\n")
        for name in attendance:
            present_report_file.write(f"{name} - {current_time}\n")

    print(f"Present attendance report saved: {present_report_path}")

    # Check for absentees not captured on the camera
    absent_not_captured = set(face_names) - set(attendance)
    
    # Folder for absent reports
    absent_report_folder = os.path.join(attendance_report_folder, "absent_reports")
    os.makedirs(absent_report_folder, exist_ok=True)
    absent_report_path = os.path.join(absent_report_folder, f"absent_report_{current_time}.txt")
    with open(absent_report_path, "w") as absent_report_file:
        absent_report_file.write(f"Absent and not captured:\n")
        for name in absent_not_captured:
            # You might want to add more details or actions for absentees
            print(f"{name} was absent and not captured. Saving details to {absent_report_path}")
            absent_report_file.write(f"{name} was absent - {current_time}\n")

def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []

    for i, name in enumerate(face_names):
        face = fr.load_image_file(f"{faces_path}\\{name}")
        face_encodings.append(fr.face_encodings(face)[0])
        face_names[i] = name.split(".")[0]  # To remove ".jpg" or any other image extension

    return face_encodings, face_names

# Retrieving face encodings and storing them in the face_encodings variable, along with the names
face_encodings, face_names = get_face_encodings()

# Reference to webcam
video = cv2.VideoCapture(0)

# Setting variable which will be used to scale size of image
scl = 2

# Confidence threshold for face recognition
confidence_threshold = 50.0  # Set your desired confidence threshold

# List to track attendance
attendance = []

# Continuously capturing webcam footage
while True:
    success, image = video.read()

    # Making current frame smaller so the program runs faster
    resized_image = cv2.resize(image, (int(image.shape[1] / scl), int(image.shape[0] / scl)))

    # Converting current frame to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Retrieving face location coordinates and unknown encodings
    face_locations = fr.face_locations(rgb_image)
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    # Iterating through each encoding, as well as the face's location
    for face_encoding, face_location in zip(unknown_encodings, face_locations):
        # Comparing known faces with unknown faces
        face_distances = fr.face_distance(face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        match_percentage = (1 - face_distances[best_match_index]) * 100

        # If match percentage is below the confidence threshold, classify as unknown
        if match_percentage < confidence_threshold:
            name = unknown
        else:
            name = face_names[best_match_index]
            mark_attendance(name)

        # Setting coordinates for face location
        top, right, bottom, left = face_location

        # Drawing rectangle around face
        cv2.rectangle(image, (left * scl, top * scl), (right * scl, bottom * scl), (0, 0, 255), 2)

        # Setting font and displaying text of name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{name} ({match_percentage:.2f}%)", (left * scl, bottom * scl + 20), font, 0.8, (255, 255, 255), 1)

    # Displaying the final image on the screen
    cv2.imshow("frame", image)
    key = cv2.waitKey(1)
    if key == 27:
        # Save attendance report and exit when 'Esc' key is pressed
        save_attendance_report()
        break

cv2.destroyAllWindows()
video.release()