import cv2
import tensorflow as tf
import dlib
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variables
api_key = os.getenv("YOUR_API_KEY")

# Check if the API key is loaded correctly
if not api_key:
    print("Error: API key not found in .env file.")
    exit()
else:
    print("API key loaded successfully!")  # For debugging purposes

# Load models (adjust paths if needed)
try:
    # Ensure these paths are correct relative to the location of this script
    emotion_model = tf.keras.models.load_model("models/video.h5")
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("models/face_landmarks.dat")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Example: Load an image and do face detection
image_path = "test_image.jpg"  # Replace with a real image path
if os.path.exists(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for i, face in enumerate(faces):  # Enumerate to get face index
        print(f"Face {i + 1} detected!")  # Print face number

        # Get facial landmarks
        landmarks = landmark_predictor(gray, face)
        # You can now use 'landmarks' for further analysis

        # Example: Draw a rectangle around the face (for visualization)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    cv2.imshow("Face Detection", img)  # Display the image with rectangles
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window
else:
    print(f"Error: Image '{image_path}' not found.")