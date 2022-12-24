import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2

import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.models import model_from_yaml
from keras.optimizers import RMSprop, SGD

# Import the backend
from keras import backend as K

class Fruite_Quality_Detection():
    def __init__(self, Train_Dir="train",Test_Dir="test" ,Model_Dir="model_cnn.yaml"):
        self.train_dir = Train_Dir  # Train Directory
        self.test_dir = Test_Dir # Test Directory
        self.model_dir =Model_Dir #Model Directory
        print(os.listdir(Train_Dir))
        print(os.listdir(Test_Dir))

    def Train_Image_Acquisition(self):
        fruit_images = []
        labels = []
        train_dir = self.train_dir
        for fruit_dir_path in glob.glob(train_dir+"/*"):
            fruit_label = fruit_dir_path.split("/")[-1]
            for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (110, 110))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # why do we need to convert the RGB2BGR
                # I don't think it is going to affect training
                # BGR was a choice made for historical reasons and now we have to live with it. In other words, BGR is the horse’s ass in OpenCV.
                fruit_images.append(image)
                labels.append(fruit_label)
        fruit_images = np.array(fruit_images)
        labels = np.array(labels)
        label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
        id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
        label_ids = np.array([label_to_id_dict[x] for x in labels])
        self.fruit_images = fruit_images
        self.labels = labels
        self.label_ids =label_ids
        print("------Train Image Aqucisition--------- ")
        print(fruit_images)
        print(labels)
        print(label_to_id_dict)
        print(label_ids)

    def Test_Image_Acquisition(self):
        validation_fruit_images = []
        validation_labels = []
        test_dir = self.test_dir

        for fruit_dir_path in glob.glob(test_dir+"/*"):
            fruit_label = fruit_dir_path.split("/")[-1]
            for image_path in glob.glob(os.path.join(fruit_dir_path, "*.png")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (110, 110))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # why do we need to convert the RGB2BGR
                # I don't think it is going to affect training
                # BGR was a choice made for historical reasons and now we have to live with it. In other words, BGR is the horse’s ass in OpenCV.
                validation_fruit_images.append(image)
                validation_labels.append(fruit_label)
        validation_fruit_images = np.array(validation_fruit_images)
        validation_labels = np.array(validation_labels)
        label_to_id_dict = {v: i for i, v in enumerate(np.unique(validation_labels))}
        id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
        validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])
        self.validation_fruit_images =validation_fruit_images
        self.validation_labels=validation_labels
        self.validation_label_ids=validation_label_ids
        print("------Test Image Aqucisition--------- ")
        print(validation_fruit_images )
        print(validation_labels)


    def Data_Preprocessing(self):
        X_train, X_test = self.fruit_images, self.validation_fruit_images
        Y_train, Y_test = self.label_ids, self.validation_label_ids

        # Normalize color values to between 0 and 1
        X_train = X_train / 255
        X_test = X_test / 255

        # Make a flattened version for some of our models
        X_flat_train = X_train.reshape(X_train.shape[0], 110 * 110 * 3)
        X_flat_test = X_test.reshape(X_test.shape[0], 110 * 110 * 3)

        # One Hot Encode the Output what is this 60
        Y_train = keras.utils.to_categorical(Y_train, 60)
        Y_test = keras.utils.to_categorical(Y_test, 60)

        print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        print('Flattened:', X_flat_train.shape, X_flat_test.shape)
        print(X_train[200].shape)
        plt.imshow(X_train[0])
        plt.show()
        self.X_train=X_train
        self.X_test=X_test
        self.X_flat_train=X_flat_train
        self.X_flat_test=X_flat_test
        self.Y_train =Y_train
        self.Y_test = Y_test
        print("------Data Preprocessing and Analysis--------- ")
        print(X_train )
        print(X_test)
        print(X_flat_train)
        print(X_flat_test)
        print(Y_train)
        print(Y_test)


    def Train_NN(self):
        X_train=self.X_train
        X_test=self.X_test
        X_flat_train=self.X_flat_train
        X_flat_test=self.X_flat_test
        Y_train= self.Y_train
        Y_test =self.Y_test
        model_dense = Sequential()

        # Add dense layers to create a fully connected MLP
        # Note that we specify an input shape for the first layer, but only the first layer.
        # Relu is the activation function used
        model_dense.add(Dense(128, activation='relu', input_shape=(X_flat_train.shape[1],)))
        # Dropout layers remove features and fight overfitting
        model_dense.add(Dropout(0.1))
        model_dense.add(Dense(64, activation='relu'))
        model_dense.add(Dropout(0.1))
        # End with a number of units equal to the number of classes we have for our outcome
        model_dense.add(Dense(60, activation='softmax'))
        model_dense.summary()
        # Compile the model to put it all together.
        # categorical_crossentropy loss
        model_dense.compile(loss='categorical_crossentropy',
                            optimizer=RMSprop(),
                            metrics=['accuracy'])

        # Changed the batch size here
        history_dense = model_dense.fit(X_flat_train, Y_train,
                                        batch_size=50,
                                        epochs=10,
                                        verbose=1,
                                        validation_data=(X_flat_test, Y_test))
        score = model_dense.evaluate(X_flat_test, Y_test, verbose=0)
        print("------Neural Networks Result--------- ")
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def Train_Relu(self):
        X_train=self.X_train
        X_test=self.X_test
        X_flat_train=self.X_flat_train
        X_flat_test=self.X_flat_test
        Y_train= self.Y_train
        Y_test =self.Y_test

        model_deep = Sequential()
        # Add dense layers to create a fully connected MLP
        # Note that we specify an input shape for the first layer, but only the first layer.
        # Relu is the activation function used
        model_deep.add(Dense(256, activation='relu', input_shape=(X_flat_train.shape[1],)))
        # Dropout layers remove features and fight overfitting
        model_deep.add(Dropout(0.05))
        model_deep.add(Dense(128, activation='relu'))
        model_deep.add(Dropout(0.05))
        model_deep.add(Dense(128, activation='relu'))
        model_deep.add(Dropout(0.05))
        model_deep.add(Dense(128, activation='relu'))
        model_deep.add(Dropout(0.05))
        model_deep.add(Dense(128, activation='relu'))
        model_deep.add(Dropout(0.05))
        # End with a number of units equal to the number of classes we have for our outcome
        model_deep.add(Dense(60, activation='softmax'))

        model_deep.summary()

        # Compile the model to put it all together.
        model_deep.compile(loss='categorical_crossentropy',
                           optimizer=RMSprop(),
                           metrics=['accuracy'])

        history_deep = model_deep.fit(X_flat_train, Y_train,
                                      batch_size=50,
                                      epochs=10,
                                      verbose=1,
                                      validation_data=(X_flat_test, Y_test))
        score = model_deep.evaluate(X_flat_test, Y_test, verbose=0)
        print("------Neural Network Relu --------- ")
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def Train_CNN(self):
        X_train=self.X_train
        X_test=self.X_test
        X_flat_train=self.X_flat_train
        X_flat_test=self.X_flat_test
        Y_train= self.Y_train
        Y_test =self.Y_test
        # The network results in very less accuracy if you are using only RELU layers as it is unable to capture the features.
        # there are using maxpool convolution and final dense layer.
        model_cnn = Sequential()
        # First convolutional layer, note the specification of shape
        model_cnn.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=(110, 110, 3)))
        model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
        model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
        model_cnn.add(Dropout(0.25))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(128, activation='relu'))
        model_cnn.add(Dropout(0.5))
        model_cnn.add(Dense(60, activation='softmax'))

        model_cnn.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

        model_cnn.fit(X_train, Y_train,
                      batch_size=128,
                      epochs=1,
                      verbose=1,
                      validation_data=(X_test, Y_test))
        score = model_cnn.evaluate(X_test, Y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model_cnn.fit(X_train, Y_train,
                      batch_size=128,
                      epochs=30,
                      verbose=1,
                      validation_data=(X_test, Y_test))
        score = model_cnn.evaluate(X_test, Y_test, verbose=0)
        print("------Convoluational Neural Networks Result--------- ")
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        #fname = "/Users/sriramreddy/Downloads/523/ex2/weights_cnn_classification_augmented_data.hdf5"
        #fname =self.Model_Dir
        #model_cnn.save_weights(fname, overwrite=True)
        # In future, you can use this model and later you can load this model for prediction
        # serialize model to YAML
        model_yaml = model_cnn.to_yaml()
        model_dir=self.model_dir
        with open(model_dir, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model_cnn.save_weights("model_cnn.h5")
        print("Saved model to disk")

    def Load_CNN_Info(self,):
        X_train = self.X_train
        X_test = self.X_test
        X_flat_train = self.X_flat_train
        X_flat_test = self.X_flat_test
        Y_train = self.Y_train
        Y_test = self.Y_test
        # later...
        model_dir = self.model_dir
        # load YAML and create model
        yaml_file = open(model_dir, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        loaded_model.load_weights("model_cnn.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        score = loaded_model.evaluate(X_test, Y_test, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    def CNN_test_Image(self,Image_test):
        import numpy as np
        from keras.preprocessing import image
        # later...
        model_dir = self.model_dir
        # load YAML and create model
        yaml_file = open(model_dir, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        test_image = image.load_img(Image_test, target_size=(110, 110))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        if result[0][0] == 1:
            prediction = 'Fresh Apple'
        elif result[0][1] == 1:
            prediction = 'Fresh Banana'
        elif result[0][2] == 1:
            prediction = 'Fresh Orange'
        elif result[0][3] == 1:
            prediction = 'Rotten Apple'
        elif result[0][4] == 1:
            prediction = 'Rotten Banana'
        else:
            prediction = 'Rotten Oranges'
        print("This fruit is",prediction)



if __name__ == "__main__":
    Fruite_Quality_Detection= Fruite_Quality_Detection()
    Fruite_Quality_Detection.Train_Image_Acquisition()
    Fruite_Quality_Detection.Test_Image_Acquisition()
    Fruite_Quality_Detection.Data_Preprocessing()
    #Fruite_Quality_Detection.Train_NN()
    #Fruite_Quality_Detection.Train_Relu()
    Fruite_Quality_Detection.Train_CNN()
    #Fruite_Quality_Detection.Load_CNN_Info()
    #Image_test_dir=""
    #Fruite_Quality_Detection.CNN_test_Image(Image_test_dir)