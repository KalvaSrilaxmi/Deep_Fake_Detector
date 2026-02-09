import cv2
import os

# --- Configuration ---
VIDEO_INPUT_DIR = 'data/Real/'  # Contains your Real video files
IMAGE_OUTPUT_DIR = 'data/Real_Frames/' # Directory to save the extracted face images
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FRAME_SKIP = 10 # Process only 1 frame every X frames (e.g., 10 to reduce size)
IMG_SIZE = 128

# --- Setup ---
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if not os.path.exists(IMAGE_OUTPUT_DIR):
    os.makedirs(IMAGE_OUTPUT_DIR)

if face_cascade.empty():
    print("ERROR: Could not load face cascade XML file.")
    exit()

print(f"Starting frame and face extraction from videos in {VIDEO_INPUT_DIR}...")
total_faces_saved = 0

for video_name in os.listdir(VIDEO_INPUT_DIR):
    if video_name.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(VIDEO_INPUT_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to reduce dataset size
            if frame_count % FRAME_SKIP == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Sort faces and take the largest one
                    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                    
                    # Crop the face
                    face_crop = frame[y:y+h, x:x+w]
                    # Resize to model input size (128x128)
                    face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                    
                    # Save the cropped face image
                    frame_filename = f"{video_name[:-4]}_face_{frame_count}.jpg"
                    output_path = os.path.join(IMAGE_OUTPUT_DIR, frame_filename)
                    cv2.imwrite(output_path, face_resized)
                    total_faces_saved += 1

        cap.release()
        print(f" -> Processed {video_name}. Saved {total_faces_saved} faces so far.")

print(f"\n✅ All video processing complete. Total faces saved: {total_faces_saved}")