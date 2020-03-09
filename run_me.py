#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import cv2
import numpy as np
from utils import *
from termcolor import colored


# Limit Tensorflow messages to errors
tf.logging.set_verbosity(tf.logging.WARN)

"""This is the main script. It performs hand segmentation and sign classification"""

# Creating relevant objects
substractor = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)

# Build the CNN model
NUM_CATEGORIES = len(POSTURE_CLASSES)
sign_model = create_posture_model(NUM_CATEGORIES)
sign_model.load_weights('Classifiers/posture_model_weights.h5')

# Read calibration thresholds
try:
    t_min = np.load('Data/r_min.npy')
    t_max = np.load('Data/r_max.npy')
except IOError:
    t_min = 10
    t_max = 50
    print('Using default segmentation thresholds. You might need to perform calibration with script ',
          colored('skin_calibration.py', 'green'))
    print('\n')

# Declaring initialization variables
_, previous_frame = cap.read()
isDetected = True
isMoving = False
count_frames = 0 
label = None

print('The program will start. {} to Quit \n \nLabels : \n'.format(colored('Press Q', 'red')))

while cap.isOpened():

    ret, img = cap.read()
    
    # We perform hand segmentation
    hand, isDetected, mask = extract_hand(img, isDetected, t_min, t_max)
    
    if hand is not None:
        # we determine if there is movement based on the comparison of the current frame and 
        # the frame that was 5 frames before
        if count_frames % 5 == 0:
            isMoving = get_motion(img, previous_frame, substractor)
            previous_frame = img
            count_frames = 0
        else:
            count_frames += 1

        # Based on whether there is motion or not, image is sent to the right CNN
        if isMoving:
            label = 'This is where we use motion classifier'
        else:
            label = get_posture_label(hand, sign_model, posture_classes)
        print(label)

    else:
        hand = np.zeros((64, 64))
        pass
    
    cv2.imshow('hand', hand)
    cv2.imshow('input', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
