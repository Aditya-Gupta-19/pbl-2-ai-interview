from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import dlib
from imutils import face_utils
from scipy.spatial import distance

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
try:
    # Make sure you have these model files in the correct location
    emotion_model = load_model('Models/video.h5')
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("Models/face_landmarks.dat")
    
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    # You might want to exit here if models can't be loaded
    # import sys
    # sys.exit(1)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def eye_aspect_ratio(eye):
    # Calculate the euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Calculate the euclidean distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/api/analyze_face', methods=['POST'])
def analyze_face():
    # Get image data from request
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    # Decode base64 image
    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error decoding image: {str(e)}'}), 400
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    faces = face_detector(gray, 1)
    
    results = []
    
    for (i, face) in enumerate(faces):
        # Get facial landmarks
        shape = landmark_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Get face coordinates
        (x, y, w, h) = face_utils.rect_to_bb(face)
        
        # Extract face for emotion prediction
        face_img = gray[y:y+h, x:x+w]
        
        # Resize face to match model input
        try:
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)
            
            # Make prediction
            prediction = emotion_model.predict(face_img)[0]
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]
            confidence = float(prediction[emotion_idx])
            
            # Extract eye coordinates for blink detection
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Calculate eye aspect ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Determine if eyes are closed
            eyes_closed = ear < 0.2
            
            # Add results for this face
            results.append({
                'face_id': i,
                'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'emotion': emotion,
                'confidence': float(confidence),
                'eyes_closed': bool(eyes_closed),
                'ear': float(ear),
                'emotions': {label: float(pred) for label, pred in zip(emotion_labels, prediction)}
            })
            
        except Exception as e:
            print(f"Error processing face: {str(e)}")
            continue
    
    return jsonify({
        'faces': results,
        'count': len(results)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)