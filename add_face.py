import cv2
import os

def capture_face(name):
    """
    Utility to capture a reference face image for the system.
    """
    faces_dir = 'faces'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Position yourself for face capture: {name}")
    print("Press 's' to Save or 'q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow("Face Capture Utility", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filepath = os.path.join(faces_dir, f"{name.lower()}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"Face saved as {filepath}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_face(name)
