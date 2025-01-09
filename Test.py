import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')  # The best model weights after training

# Start video capture (0 for webcam or provide the path to a video file)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference on the current frame
    results = model.predict(frame, save=False)  # Perform prediction

    # Visualize results on the frame
    annotated_frame = results[0].plot()  # Plot the bounding boxes and labels on the frame

    # Display the frame with bounding boxes
    cv2.imshow("Helmet Detection", annotated_frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
