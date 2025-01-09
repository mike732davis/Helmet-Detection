# Helmet-Detection
Helmet Detection with YOLOV8

# Dataset Link: 
https://www.kaggle.com/datasets/andrewmvd/helmet-detection/data

# Packages:
	python - 3.10.11
	xmltodict - 0.14.2
	opencv-python -  4.10.0.84
	scikit-learn - 1.1.2
	ultralytics - 8.3.58

# Workflow:
1) Download Dataset from Kaggle
2) Create YOLO file for annotation from PASCAL VOC file (Annotation folder files) using packages xmltodict
	opencv-python (vocToYolo.py)
3) Split the dataset as train and test set train 80% and test 20% using scikit-learn (trainTestSplit.py)
4) Train YOLOV8 model for helmet detection using ultralytics and OpenCV (Training.py)
	note:
	change the path of train and val with the path in your system
5) After training you will get a .yaml file and best.pt file (Test.py)
6) Load the best.pt file after training to the test and video frame will be opened using open-cv (Test.py)
7) if you want to check with image for testing you can use TestImg.py
