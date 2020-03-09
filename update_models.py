from utils import *
import keras
from keras.utils import plot_model
import pydot
import graphviz

# First process the posture datasets and update model parameters if needed
print('Processing posture dataset... \n')
retrain = process_dataset_dir('Datasets')
process_dataset('Test_Data/Johann', 'test_data.npy', 'test_labels.npy')

# Retrain model if some data was added
if retrain:

    print('New dataset : Updating model weights')
    train_posture_model(1)

else:
    print('Data and model are up to date \n')


# Train the gesture model
process_dataset('Dataset_gesture/gesture_train', 'gesture_train_data.npy', 'gesture_train_labels.npy')
process_dataset('Dataset_gesture/gesture_validation', 'gesture_val_data.npy', 'gesture_val_labels.npy')

gesture_train_data = np.load('Data/gesture_train_data.npy')
gesture_train_labels = np.load('Data/gesture_train_labels.npy')

gesture_val_data = np.load('Data/gesture_val_data.npy')
gesture_val_labels = np.load('Data/gesture_val_labels.npy')

train_data_len = len(gesture_train_data)
p_train = np.random.permutation(train_data_len)
gesture_train_data = gesture_train_data[p_train]
gesture_train_labels = gesture_train_labels[p_train]

val_data_len = len(gesture_val_data)
p_val = np.random.permutation(val_data_len)
gesture_val_data = gesture_val_data[p_val]
gesture_val_labels = gesture_val_labels[p_val]

print(gesture_val_labels)

gesture_model = create_gesture_model()
plot_model(gesture_model, to_file='gesture_model.png')
gesture_history = gesture_model.fit(gesture_train_data, gesture_train_labels,
                  validation_data=(gesture_val_data, gesture_val_labels),
                  batch_size=64, epochs=30, shuffle=True, verbose=1)

print('Saving trained model weights... \n')
gesture_model.save_weights('Classifiers/gesture_model_weights.h5')

plot_history(gesture_history)

# Confusion matrix
val_predictions = gesture_model.predict_classes(gesture_val_data, batch_size=32, verbose=0)
cm = confusion_matrix(gesture_val_labels, val_predictions)

plt.figure()
plot_confusion_matrix(cm, GESTURE_CLASSES, title='Confusion Matrix', normalize=True)
plt.show()
