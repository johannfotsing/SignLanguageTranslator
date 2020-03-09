#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from termcolor import colored

MIN_MAX = 200
MAX_MAX = 300
SCALE = 100

# Avoid divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')


def nothing(x):
    pass


def skin_calibration(__cap):

    print('Stand in front of the camera and raise one hand.\nSet trackbars so that only your skin is white\
     and everything else is dark.\nThen {} when you are satisfied with the result.'.format(colored('Press C', 'green')))
    print('You can {} to quit without saving.\n'.format(colored('Press Q', 'red')))

    ret, frame = __cap.read()
    cv2.imshow('Mask', np.zeros(frame.shape))
    min_v = 10
    max_v = 300
    cv2.createTrackbar('Background', 'Mask', min_v, MIN_MAX, nothing)
    cv2.createTrackbar('Skin', 'Mask', max_v, MAX_MAX, nothing)

    quit_calibration = False

    while True:

        ret, frame = __cap.read()

        ratio = frame[:, :, 2]/frame[:, :, 1]
        mask = np.array(1*np.logical_and(min_v/SCALE <= ratio, max_v/SCALE >= ratio), dtype=np.uint8)

        # Blurring using a Gaussian filter to get rid of noise
        blurred_mask = cv2.GaussianBlur(mask, (9, 9), 0)
        mask = np.logical_and(blurred_mask[:, :] > 0, 1)   # All non-zero values become 1
        mask = np.uint8(mask)   # Casting to uint8

        cv2.imshow('Mask', mask*255)

        # Update ratio min_ratio and max_ratio
        min_v = cv2.getTrackbarPos('Background', 'Mask')
        max_v = cv2.getTrackbarPos('Skin', 'Mask')

        # Press c when calibration is done
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_calibration = True
            break

    cv2.destroyAllWindows()

    # No save if quited
    if quit_calibration:
        print('Quited calibration before saving.')
        return

    # Save the calibration parameters
    np.save('Data/r_max.npy', max_v/SCALE)
    np.save('Data/r_min.npy', min_v/SCALE)

    print('Calibration done !')

    return


cap = cv2.VideoCapture(0)
skin_calibration(cap)
