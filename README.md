# Smile Detector
A real-time smile detector program

## How to Run

Pre-requisite: python and pip installed

Install the requirements 
```bash
sudo pip install -r requirements.txt
```

Run the smile detection using CNN
```bash
python detect_smile.py --cascade cascade/haarcascade_frontalface_default.xml --model model/output/lenet.hdf5
```

Run the smile detection using Haar cascade
```bash
python detect_smile_cascade.py
```




## Description
Main program captures real-time video from webcam. Then, detects faces and draw a rectangle around the face area. When the person smiles, it should print a text "Smiling" above the rectangle area. When not smiling, it should print a text "Not Smiling".

## How it works
* Get smiling and not smiling dataset images;
* Train a network in the dataset;
* Evaluate the network;
* Detect face with Haar Cascade;
* Extract the Region of Interest (ROI);
* Pass the ROI through trained network;
* Output the result from trained network.

## Project structure
cascade/  - Folder for cascade classifiers. Provide any classifiers here.

dataset/  - Data for fitting model

models/   - Script to train model and output for trained model


# Reference
https://www.pyimagesearch.com/2021/07/14/smile-detection-with-opencv-keras-and-tensorflow/