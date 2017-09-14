from keras.models import Model
import matplotlib.pyplot as plt
import pickle


file = open("history_object.obj",'rb')
history_object = pickle.load(file)

### print the keys contained in the history object
print(history_object.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object['loss'])
plt.plot(history_object['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
