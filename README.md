Face Recognition System

Overview

This project implements a face recognition system using OpenCV and the LBPH algorithm. The system detects faces, captures images, trains a model, and recognizes faces in real-time.

Process Flow

Face Detection & Dataset Creation: Captures and stores face images.

Model Training: Uses the LBPHFaceRecognizer to train a recognition model.

Face Recognition: Identifies faces from a live webcam feed.

Requirements

Python 3.x

OpenCV

NumPy

Pillow (PIL)

Setting Up the Project in VS Code

1. Install Dependencies

Ensure you have Python installed, then run:

pip install opencv-python opencv-contrib-python numpy pillow

2. Create and Navigate to Your Project Folder

mkdir face-recognition
cd face-recognition

3. Clone or Create Project Files

Save the following scripts in your folder:

dataset_creator.py (Captures face images)

trainer.py (Trains the model)

recognizer.py (Recognizes faces)

haarcascade_frontalface_default.xml (Face detection model)

4. Run the Scripts

a) Capture Face Dataset:

python dataset_creator.py

Enter a unique ID when prompted.

The script captures 300 images and saves them in datasets/.

b) Train the Model:

python trainer.py

Generates a Trainer.yml file for recognition.

c) Recognize Faces:

python recognizer.py

Press a to exit recognition mode.

File Structure

face-recognition/
│── datasets/            # Captured face images
│── Trainer.yml          # Trained model
│── dataset_creator.py   # Captures face images
│── trainer.py           # Trains the model
│── recognizer.py        # Recognizes faces
│── haarcascade_frontalface_default.xml  # Face detection model
│── README.md            # Documentation

Notes

Update name_list in recognizer.py to map IDs to names.

Make sure haarcascade_frontalface_default.xml is in your project folder.

Capture varied images for better recognition accuracy.

License

MIT License

Author

[Your Name]
