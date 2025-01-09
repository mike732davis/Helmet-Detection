import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# Path to validation images
val_images_path = "helmetDataset/test"  # Update this with the correct path

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Ensure the best.pt is correctly located

# List all image files in the validation directory
image_files = list(Path(val_images_path).rglob("*.png")) + list(Path(val_images_path).rglob("*.jpg"))

if not image_files:
    print("No images found in the validation directory.")
else:
    # Iterate through validation images
    for image_file in image_files:
        # Read the image
        img = cv2.imread(str(image_file))

        # Perform inference on the image
        results = model.predict(img, save=False)

        # Visualize results on the image
        annotated_img = results[0].plot()

        # Display the annotated image
        cv2.imshow("Helmet Detection - Validation Image", annotated_img)

        print(f"Processed: {image_file}")

        # Wait for a key press to proceed to the next image or exit
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
