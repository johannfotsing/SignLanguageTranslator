#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Use this script to create datasets."""

# Imports
import os
import cv2
import numpy as np
import time
from termcolor import colored
from utils import extract_hand


# Message at start up of the script
print('\n --------------- BUILDING OF DATASET ---------------\n')
print('Hi, this scripts helps you build a dataset for the sign language translator.\n\
 If you want to add signs to an already existing dataset, you can enter its name in the next input field. \n\
 If you want to build a dataset from scratch, enter the name of your new dataset in the next input field. \n')

# Read calibration thresholds
try:
    t_min = np.load('Data/r_min.npy')
    t_max = np.load('Data/r_max.npy')
except IOError:
    t_min = 1.15
    t_max = 4
    print('Using default segmentation thresholds. You might need to perform calibration with script ',
          colored('skin_calibration.py', 'green'))

# User enters some parameters
dataSetName = input("Please enter dataset name: ")
numberOfSigns = input("Please enter the number of signs you want to add: ")
numberOfInstances = input("Please enter the number of images that you want for a given class: ")

cap = cv2.VideoCapture(0)
print("\nMake your first sign and get ready for the recording ! \n")
time.sleep(3)   # Wait for 3 seconds before starting

for i in range(int(numberOfSigns)):

    j = 0
    label = input("Please enter a label for the sign you want to add: ")
    folder_path = 'Datasets/'+dataSetName+'/'+label
    os.makedirs(folder_path)
    detected = True
    while j < int(numberOfInstances)//2:

        # Create file path.
        filepath1 = folder_path+'/'+str(2*j)+'.png'
        filepath2 = folder_path+'/'+str(2*j+1)+'.png'

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Hand extraction in a 64x64 image
        hand, detected, mask = extract_hand(frame, detected, t_min, t_max)
        
        # Storage of the image
        if hand is not None:
            cv2.imwrite(filepath1, hand)
            cv2.imwrite(filepath2, np.flip(hand, axis=1))
            j += 1

        else:
            hand = np.zeros((64, 64))

        # Display the resulting frame and the hand extraction hand image
        cv2.imshow('Frame', frame)
        cv2.imshow('Hand extracted', hand)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
