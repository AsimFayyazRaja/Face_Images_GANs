from __future__ import print_function, division

import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
import random
from random import randint
from sklearn.metrics import average_precision_score
import pandas as pd
from scipy import misc as cv2
import glob
import tensorflow as tf
from PIL import Image
import image
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage import img_as_float
import cv2 as cv
from skimage import transform
import matplotlib.image as mpimg

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0004, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(624 * 2 * 2, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((2, 2, 624)))
        model.add(UpSampling2D())
        
        model.add(Conv2D(512, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        #4x4x512
        model.add(Conv2D(256, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        #8x8x256
        model.add(Conv2D(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        #16x16x128
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        #32x32x64
        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))
        #64x64x3
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    
    def train(self, epochs, batch_size, save_interval):

        #loading existing model to resume training
        self.generator=load_model('gen_model.h5')
        self.discriminator=load_model('disc_model.h5')

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()
        epochs+=2
        X_train=get_data()
        X_train=np.array(X_train)
        print(X_train.shape)
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        
        #X_train = np.expand_dims(X_train, axis=3)
        #print(X_train.shape)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            #print("SHAPE: ",imgs.shape)
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            #disc loss:-
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)
            #gen loss:-


            # Plot the progress
            print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            if epoch+1>=epochs:
                print("saving model..")
                self.discriminator.save('disc_model.h5')
                self.generator.save('gen_model.h5')
                print("model saved")


    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0]) #,cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/new-face_%d.png" % epoch)
        plt.close()

def get_data():
        folder="data"
        r=0
        print("Getting images for training..")
        training_data = []
        bag=[]
        with tqdm(total=len(glob.glob(folder+"/*.jpg"))) as pbar:
            for img in glob.glob(folder+"/*.jpg"):
                temp=[]
                n= np.array(cv2.imread(img))
                if n.shape==(64,64):
                    continue
                #n= cv.normalize(n,n, 0, 255, cv.NORM_MINMAX)
                #n= n/(n.max()/255.0)
                bag.append(n)
                pbar.update(1)
                r+=1
        return bag

def makeImages():
    model=load_model('gen_model.h5')
    noise = np.random.normal(0, 1, (1, 100))
    gen_imgs = model.predict(noise)
    gen_imgs=gen_imgs[0]
    gen_imgs = 0.5 * gen_imgs + 0.5
    #print(gen_imgs.shape)
    gen_imgs=np.array(gen_imgs)
    plt.imsave('test4.png',gen_imgs)
    fig=plt.figure(figsize=(4, 4))
    rows=1
    coloumns=3
    for i in range(1, 1*3 +1):
        img=mpimg.imread("test"+str(i)+".png")
        fig.add_subplot(rows, coloumns, i)
        plt.imshow(img)
    plt.show()
    

if __name__ == '__main__':
    makeImages()
    makeImages()
    '''
    dcgan = DCGAN()
    dcgan.train(epochs=6000, batch_size=64, save_interval=20)
    '''
    
