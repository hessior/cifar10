import keras
from keras.layers import Input
from keras.layers import Conv2D, GlobalAveragePooling2D, Activation
from keras.models import Model
from keras import backend as K
from keras.datasets import cifar10
import keras.utils
import tensorflow as tf
import numpy as np
import os


def prepare_data(data_path, use_build_in=False):
    if use_build_in == False:
        x_train = np.load(os.path.join(data_path,'cifar10_x_train.npy'))
        x_train = x_train.reshape((-1,3,32,32)).swapaxes(1,2).swapaxes(2,3)    ## (50000,32,32,3)
        x_test = np.load(os.path.join(data_path,'cifar10_x_test.npy'))
        x_test = x_test.reshape((-1,3,32,32)).swapaxes(1,2).swapaxes(2,3)      ## (10000,32,32,3)
        x_train = x_train / 255.
        x_test = x_test / 255.
        #_mean = np.mean(x_train, axis=(1,2))[:,None,None,:]
        #_std = np.std(x_train, axis=(1,2))[:,None,None,:]
        #x_train = (x_train - _mean) / _std
        #_mean = np.mean(x_test, axis=(1,2))[:,None,None,:]
        #_std = np.std(x_test, axis=(1,2))[:,None,None,:]
        #x_test = (x_test - _mean) / _std
        y_train = np.load(os.path.join(data_path,'cifar10_y_train.npy'))   ## (50000,)
        y_train = keras.utils.to_categorical(y_train, 10)                  ## (50000,10)
        y_test = np.load(os.path.join(data_path,'cifar10_y_test.npy'))    ## (10000,)
        y_test = keras.utils.to_categorical(y_test, 10)                    ## (10000,10)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return (x_train, y_train, x_test, y_test)
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        return (x_train, y_train, x_test, y_test)


class Cifar10_model:
    
    def __init__(self, data_path, ckpt_name="cifar10_ckpt"):
        self.data_path = data_path
        self.img, self.label, self.test_img, self.test_label = prepare_data(data_path, use_build_in=False)
        if not os.path.isdir(ckpt_name):
            os.mkdir(ckpt_name)
        self.ckpt_path = os.path.join(os.getcwd(),ckpt_name)
        
    def build(self, weight_name=None):
        self.inpu = Input(shape=(32,32,3))               ## 4-dim, need shape
        self.conv1 = Conv2D(96, (3,3), padding='same', activation='relu')(self.inpu)
        self.conv2 = Conv2D(96, (3,3), padding='same', activation='relu')(self.conv1)
        self.conv3 = Conv2D(96, (3,3), strides=(2,2), padding='same',activation='relu')(self.conv2)
        self.conv4 = Conv2D(192, (3,3), padding='same', activation='relu')(self.conv3)
        self.conv5 = Conv2D(192, (3,3), padding='same', activation='relu')(self.conv4)
        self.conv6 = Conv2D(192, (3,3), strides=(2,2), padding='same', activation='relu')(self.conv5)
        self.conv7 = Conv2D(192, (3,3), padding='valid', activation='relu')(self.conv6)
        self.conv8 = Conv2D(192, (1,1), padding='same', activation='relu')(self.conv7)
        self.conv9 = Conv2D(10, (1,1), padding='same', activation='relu')(self.conv8)
        self.logit = GlobalAveragePooling2D()(self.conv9)
        self.output = Activation('softmax')(self.logit)            ## output.shape is (?,10)
        self.model = Model(inputs=[self.inpu], outputs=[self.output])
        if weight_name:
            self.model.load_weights(os.path.join(self.ckpt_path,weight_name))

    def train(self):
        opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
        self.model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.fit(self.img, self.label, epochs=100, batch_size=128,
                        validation_data=(self.test_img,self.test_label))
        #opt2 = keras.optimizers.rmsprop(lr=0.00001,decay=1e-6)
        #self.model.compile(optimizer=opt2,loss='categorical_crossentropy',metrics=['accuracy'])
        #self.model.fit(self.img, self.label, epochs=100, batch_size=128,
        #                validation_data=(self.test_img,self.test_label))
    
    def save_weight(self, weight_name):
        self.model.save_weights(os.path.join(self.ckpt_path,weight_name))
    
    def test(self):
        opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
        self.model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
        self.scores = self.model.evaluate(self.test_img,self.test_label)

if __name__ == '__main__':
    data_path = "data/"
    weight_name = "cifar10.h5"

    cifar10_model = Cifar10_model(data_path)      ## create model, load data
    cifar10_model.build()
    cifar10_model.train()
    cifar10_model.save_weight(weight_name)
    cifar10_model.test()                        ## use either after train() or build() 
    print(cifar10_model.scores[1])