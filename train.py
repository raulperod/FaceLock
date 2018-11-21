import random
import os
import numpy as np

import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.engine import  Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K
from input import extract_data, resize_with_pad, IMAGE_SIZE
#from keras.applications import InceptionResNetV2, VGG16, VGG19, ResNet50
#from keras.applications.inception_resnet_v2 import preprocess_input
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg19 import preprocess_input
#from keras.applications.resnet50 import preprocess_input

from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils


DEBUG_MUTE = True # Stop outputing unnecessary 

class DataSet(object):

    TRAIN_DATA = './data/train/'

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        
        images, labels = extract_data(self.TRAIN_DATA)
        labels = np.reshape(labels, [-1])
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0, 100))
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], img_channels, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
            input_shape = (img_channels, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, img_channels)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
            input_shape = (img_rows, img_cols, img_channels)

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test

class Model(object):

    FILE_PATH = './models/faces.h5'
    
    TrainEpoch = 20
    
    def __init__(self):
        self.model = None

    def check_existance(self, file_path=FILE_PATH):
        if os.path.exists(file_path):
            return True
        else:
            return False

    def build_model(self, nb_classes=2, hidden_dim=512):
        
        """ KERAS MODEL WITH IMAGENET
        self.model = Sequential()
        self.model.add(InceptionResNetV2(include_top=False, pooling='avg', weights="imagenet", input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        self.model.add(VGG19(include_top=False, pooling='avg', weights="imagenet", input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        self.model.add(VGG16(include_top=False, pooling='avg', weights="imagenet", input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # Say not to train first layer (ResNet) model. It is already trained
        self.model.layers[0].trainable = False
        self.model.summary() 
        """
        """  VGG16 
        vgg_model = VGGFace(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_classes, activation='softmax', name='fc8')(x)
        custom_vgg_model = keras.engine.Model(vgg_model.input, out)
        """
        # RESNET50
        vgg_model = VGGFace(include_top=False, model='resnet50', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(nb_classes)(x)
        out = Activation('softmax')(x)
        custom_vgg_model = keras.engine.Model(vgg_model.input, out)
        
        self.model = custom_vgg_model

        self.model.summary()

    def train(self, batch_size=32, nb_epoch=40):

        dataset = DataSet()
        dataset.read()    

        #sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

        self.model.compile(loss='categorical_crossentropy',
                            optimizer=adam,
                            metrics=['accuracy'])
        
        """
        print('Using real-time data augmentation.')
        
        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = data_generator.flow(
            dataset.X_train,
            dataset.Y_train,
            batch_size=batch_size
        )

        validation_generator = data_generator.flow(
            dataset.X_valid,
            dataset.Y_valid,
            batch_size=batch_size
        )

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=dataset.X_train.shape[0],
            epochs=nb_epoch,
            validation_data=validation_generator,
            validation_steps=dataset.X_valid.shape[0]
        )
        """

        self.model.fit(dataset.X_train, dataset.Y_train, batch_size=batch_size, epochs=nb_epoch,
                    validation_data=(dataset.X_valid, dataset.Y_valid))

        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

    def save(self, file_path=FILE_PATH):
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.model.save(file_path)
        print('Model Saved.')

    def load(self, file_path=FILE_PATH):
        self.model = load_model(file_path)
        print('Model Loaded.')

    def predict(self, predict_image):
        image = resize_with_pad(predict_image)
        image = image.astype('float32')
        image /= 255

        result = self.model.predict(np.array([image]))[0]
        whois = 0 if (result[0] > result[1]) else 1

        return whois 

if __name__ == '__main__':
    model = Model()

    model.build_model()
    
    model.train(nb_epoch=model.TrainEpoch)
    model.save()