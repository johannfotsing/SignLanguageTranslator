#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import dippykit as dip
import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.metrics import confusion_matrix


# Limit Tensorflow messages to errors
tf.logging.set_verbosity(tf.logging.ERROR)

# Dimension of the extracted hand image
DIM = (64, 64)

# OpenCV Face detector
face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_eye.xml')

# Avoid divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')

# Posture and gesture classes
POSTURE_CLASSES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
GESTURE_CLASSES = ['j', 'z']


def get_motion(img, previous_frame, substractor):

    """This function determines whether there is movement or not based on the previous frame and current image."""

    motion = substractor.apply(img)
    previous_motion = substractor.apply(previous_frame)

    motion_measure = (dip.MSE(previous_motion, motion)*1000)

    if motion_measure > 20:
        is_moving = True
    else:
        is_moving = False

    return is_moving


def extract_hand(img, detected, min_ratio, max_ratio):

    """This function extracts the hand from an image and returns the image in a 64x64 grayscale format."""

    # Segregating skin color pixels
    ratio = img[:, :, 2]/img[:, :, 1]
    mask = np.array(1*np.logical_and(min_ratio <= ratio, max_ratio >= ratio), dtype=np.uint8)

    # Blurring using a Gaussian filter to get rid of noise
    blurred_mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = np.logical_and(blurred_mask[:, :] > 0, 1)     # All non-zero values become 1
    mask = np.uint8(mask)   # Casting to uint8

    # Finding the contours in the image
    cnt_img, contours, _ = cv2.findContours(blurred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=lambda var_x: cv2.moments(var_x)['m00'])

    # Fetch the 2 largest skin-coloured areas
    cnt1 = []
    cnt2 = []
    try:
        cnt1 = contours_sorted[-1]   # Probably face contour (largest)
    except IndexError:
        pass

    # If we have any contour :
    if len(cnt1) != 0:

        contour_moments = cv2.moments(cnt1)
        x = int(contour_moments['m10']/contour_moments['m00'])
        y = int(contour_moments['m01']/contour_moments['m00'])
        _, r = cv2.minEnclosingCircle(cnt1)
        r = int(r)

        # Detection flag used to not abusively print error messages
        detected = True

    else:

        ''' No contour was detected '''
        if detected:
            print('Segmentation : {}.'.format(colored('Nothing Detected', 'red')))
            print('Please stand in front of the camera or wave your hand')
            print('Think about skin calibration \n')

        return None, False, mask

    if len(cnt2) != 0:

        '''We take a bit larger for face detection'''
        larger_r = int(1.4*r)

        try:
            gray_prob_face = cv2.cvtColor(np.uint8(img[y-larger_r:y+larger_r, x-larger_r:x+larger_r, :]), cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # If we cannot convert to gray, we return nothing
            return None, False, mask

        # We try to detect the face
        face = []
        try:
            face = face_cascade.detectMultiScale(gray_prob_face)
        except TypeError:
            # Then gray_prob_face is not empty
            pass

        # If face is not detected, then the hand is very likely larger than the face on the image :
        if len(face) == 0:
            # We do not 'try' here because in this case, we are sure the image is good since we already extracted with bigger size R
            gray_prob_hand = cv2.cvtColor(np.uint8(img[y-r:y+r, x-r:x+r, :]), cv2.COLOR_BGR2GRAY)

        # If face is detected, we take the other contour :
        else:

            # The hand is probably in the second contour :
            contour_moments = cv2.moments(cnt2)
            x = int(contour_moments['m10']/contour_moments['m00'])
            y = int(contour_moments['m01']/contour_moments['m00'])
            _, r = cv2.minEnclosingCircle(cnt2)
            r = int(r)

            try:
                gray_prob_hand = cv2.cvtColor(np.uint8(img[y-r:y+r, x-r:x+r, :]), cv2.COLOR_BGR2GRAY)
            except cv2.error:
                return None, False, mask

    # Now if we only detect one contour, it is the only probable hand in the image
    else:

        # We want to think that it is a hand :
        contour_moments = cv2.moments(cnt1)
        x = int(contour_moments['m10']/contour_moments['m00'])
        y = int(contour_moments['m01']/contour_moments['m00'])
        _, r = cv2.minEnclosingCircle(cnt1)
        r = int(r)

        try:
            gray_prob_hand = cv2.cvtColor(np.uint8(img[y-r:y+r, x-r:x+r, :]), cv2.COLOR_BGR2GRAY)
        except cv2.error:
            return None, False, mask

    ''' Now we double check that we picked a hand by trying to find an eye in the image '''
    eye = eye_cascade.detectMultiScale(gray_prob_hand)

    # If the face image reached here, we want to not consider it
    if len(eye) != 0:
        return None, False, mask

    gray_hand = cv2.resize(gray_prob_hand*mask[y-r:y+r, x-r:x+r], DIM, interpolation=cv2.INTER_LINEAR)

    return gray_hand, detected, mask


def process_dataset(data_path, data_name='data.npy', label_name='labels.npy'):

    """ This function processes images and store them into a numpy file for faster loading
        Labels is the name of the subfolder containing images of the same class """

    try:
        file = open("Data/dataset_list.txt", "x")
        file.close()
    except IOError:
        pass

    # We create a list of processed datasets
    file = open("Data/dataset_list.txt", "r")
    dataset_list = file.read().split('\n')
    file.close()

    if data_path in dataset_list:
        return False

    data_list = []
    labels_list = []
    description_list = []

    # Load data from image directory
    subdir = os.listdir(data_path)

    for i in range(len(subdir)):

        dir_path = data_path+'/'+subdir[i]
        files = os.listdir(dir_path)

        for j in range(len(files)):

            filepath = dir_path + '/' + files[j]
            img = cv2.imread(filepath, 0)
            im_x, im_y = img.shape
            img = img.reshape([im_x, im_y, 1])
            data_list.append(img)
            labels_list.append(i)

    # Convert data and labels to numpy arrays
    data_list = np.array(data_list)
    labels_list = np.array(labels_list)

    assert len(data_list) == len(labels_list)

    # Add data and labels to the numpy dataset bank
    if data_name in os.listdir('Data'):
        data = np.load('Data/'+data_name)
        labels = np.load('Data/'+label_name)
        data = np.append(data, data_list[:], axis=0)
        labels = np.append(labels, labels_list[:], axis=0)

    else:
        print('First dataset processing')
        data = data_list
        labels = labels_list
        posture_classes = description_list

    np.save('Data/'+data_name, data)
    np.save('Data/'+label_name, labels)

    # Update the datasets list file
    file = open("Data/dataset_list.txt", "a")
    file.write('{}\n'.format(data_path))
    file.close()

    return True


def process_dataset_dir(directory):

    data_path_list = os.listdir(directory)

    for j in data_path_list:

        change = False
        data_path = directory + '/' + j
        change = change or process_dataset(data_path)

    return change


def create_posture_model(nb_cat):

    """This function creates the model used for posture classification"""

    # Build the model

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(64, 64, 1), filters=16, kernel_size=3, activation=tf.nn.relu),
        # tf.keras.layers.Conv2D(input_shape=(64, 64, 1), filters=16, kernel_size=11, activation=tf.nn.relu,
        #                       kernel_initializer=tf.keras.initializers.glorot_normal, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        # tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu),
        # tf.keras.layers.Conv2D(filters=32, kernel_size=9, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal,
        #                       padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=4, padding='same'),
        # tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation=tf.nn.relu),
        # tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal,
        #                      padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        # tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        # tf.keras.layers.BatchNormalization(axis=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=192, activation=tf.nn.relu),
        # tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_normal, kernel_regularizer=tf.keras.regularizers.l1(0.1)),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=nb_cat, activation=tf.nn.softmax)
    ])

    # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-1, amsgrad=True)
    opt = tf.keras.optimizers.SGD(lr=0.001)

    # Compile the model
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def create_gesture_model():

    """This function creates the model used for gesture classification"""

    # Build the model

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(64, 64, 1), filters=16, kernel_size=5, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=5, strides=4, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(units=1, activation=tf.nn.softmax)
    ])

    # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-1, amsgrad=True)
    opt = tf.keras.optimizers.SGD(lr=0.001)

    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_posture_model(verbose=1, num_epochs=50):

    # Load data
    print('Loading data...')

    data = np.load('Data/data.npy')
    labels = np.load('Data/labels.npy')
    posture_classes = np.load('Data/posture_classes.npy')
    data_len = len(data)
    num_categories = len(posture_classes)

    # Random permutation : shuffling dataset
    p = np.random.permutation(data_len)
    data = data[p]
    labels = labels[p]

    print(data_len)

    ''' Training and validation datasets '''
    train_data = data
    train_labels = labels
    print(train_labels)
    val_data = np.load('Data/test_data.npy')
    val_labels = np.load('Data/test_labels.npy')

    print('Data loaded successfully. \n')

    ''' Create and compile model '''
    sign_model = create_posture_model(num_categories)

    ''' Training parameters '''
    batch_size = 384

    # Train the model
    history = sign_model.fit(train_data, train_labels, batch_size=batch_size, epochs=num_epochs, validation_data=(val_data, val_labels), verbose=verbose, shuffle=True)

    # Save trained model
    print('Saving trained model weights... \n')
    sign_model.save_weights('Classifiers/posture_model_weights.h5')

    ''' Plot training history '''
    plot_history(history)

    return


def get_posture_label(img, model):

    """This function returns the posture's label given the 64*64 extracted hand image"""

    # Reshape image for model feeding
    reshaped_img = np.expand_dims(img, 2)
    reshaped_img = np.expand_dims(reshaped_img, 0)

    # Make label prediction
    pred = model.predict_classes(reshaped_img)

    return POSTURE_CLASSES[pred[0]]


def plot_history(history):

    # Plot training results
    print('Generating training plots...')

    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = history.epoch

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'g-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'g-', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
