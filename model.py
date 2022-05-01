# Import all the required libraries, packages and files
import os
import sys
import keras
import tensorflow as tf
import numpy as np
from keras import backend as K
from numpy import random
import pandas as pd
import tensorflow.keras.utils as utils
from tensorflow.keras.layers import Dense, Reshape, Lambda, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU, BatchNormalization, Conv2DTranspose

class Sampling:
    @staticmethod
    def uae_encoder_sampling(args):
        #re-parametrization technique
        y_mean = args[0]
        sigma = 0.1 #for different datasets different variance can work better
        latent_dim = args[1]
        epsilon = K.random_normal(shape=(K.shape(y_mean)[0], latent_dim),
                                  mean=0, stddev = 1)
        z = epsilon * sigma + y_mean
        #z = tf.squeeze(z, axis = 0)
        return z

    @staticmethod
    def vae_encoder_sampling(args):
        z_mean, z_log_sigma = args[0], args[1]
        latent_dim = args[2]
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon


class MNIST(Sampling):
    def __init__(self, generator_model, inp = 28*28, ap_out=2, au_out=2, latent_dim = 96):
        self.generator_model = generator_model
        self.input_dim = inp
        self.latent_dim = latent_dim
        self.private_output_dim = ap_out
        self.utility_output_dim = au_out

    def generator(self):
        #defines generator part of our model which consists of encoder and decoder
        # -----------------------Create encoder -------------------------------
        inputs = keras.Input(shape=(self.input_dim,))
        h = Reshape((28,28,1))(inputs)
        h = Conv2D(32, 3, activation="relu", strides=2, padding="same")(h)
        h = Conv2D(64, 3, activation="relu", strides=2, padding="same")(h)
        h = Flatten()(h)
        h = Dense(32, activation="relu")(h)
        if self.generator_model == "UAE":
            y_mean = Dense(self.latent_dim)(h)
            y = Lambda(self.uae_encoder_sampling)((y_mean, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y])
        elif self.generator_model == "AE":
            y = Dense(self.latent_dim)(h)
            encoder = keras.Model(inputs, y)
        else:
            #returns VAE or beta-VAE model
            y_mean = Dense(self.latent_dim)(h)
            y_sigma = Dense(self.latent_dim)(h)
            y = Lambda(self.vae_encoder_sampling)((y_mean, y_sigma, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y_sigma, y])
        # -----------------------Create decoder -------------------------------
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        h = Reshape((7, 7, 64))(h)
        h = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(h)
        h = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(h)
        h = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(h)
        decoder_outputs = Reshape((self.input_dim,))(h)
        decoder = keras.Model(latent_inputs, decoder_outputs)
        # -----------------instantiate generator model --------------------------------
        if self.generator_model == "UAE":
            outputs = decoder(encoder(inputs)[1])
            return keras.Model(inputs, outputs)
        elif self.generator_model == "AE":
            outputs = decoder(encoder(inputs))
            return keras.Model(inputs, outputs)
        else:
            #returns VAE or beta-VAE model
            outputs = decoder(encoder(inputs)[2])
            return keras.Model(inputs, outputs), encoder

    def a_u_model(self):
        # utility provider model
        inputs1 = keras.Input(shape=(self.input_dim,))
        converter = Reshape((28,28,1))(inputs1)
        conv1 = Conv2D(32, kernel_size=4, activation='relu')(converter)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flat = Flatten()(pool2)
        outputs1 = Dense(self.utility_output_dim, activation = 'softmax')(flat)
        utility = keras.Model(inputs1, outputs1)
        return utility

    def a_p_model(self):
        #adversary model i.e. private inference model
        inputs2 = keras.Input(shape=(self.input_dim,))
        converter = Reshape((28,28,1))(inputs2)
        conv1 = Conv2D(32, kernel_size=4, activation='relu')(converter)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flat = Flatten()(pool2)
        outputs2 = Dense(self.private_output_dim, activation = 'softmax')(flat)
        private = keras.Model(inputs2, outputs2)
        return private


class FashionMNIST(Sampling):
    def __init__(self, generator_model, inp=28*28, ap_out=10, au_out=2, latent_dim = 96):
        self.generator_model = generator_model
        self.input_dim = inp
        self.latent_dim = latent_dim
        self.private_output_dim = ap_out
        self.utility_output_dim = au_out

    def generator(self):
        #defines generator part of our model which consists of encoder and decoder
        # -----------------------Create encoder -------------------------------
        inputs = keras.Input(shape=(self.input_dim,))
        filters = 16
        kernels = 3
        h = Reshape((28,28,1))(inputs)
        h = Conv2D(filters, 1, strides=1, padding="same")(h)
        h = LeakyReLU()(h)
        h = Conv2D(filters, kernels, strides=1, padding="same")(h)
        h = LeakyReLU()(h)
        for i in range(3):
            filters *= 2
            h = Conv2D(filters, kernels, strides=1, padding="same")(h)
            h = LeakyReLU()(h)
            h = MaxPooling2D(pool_size=(2,2))(h)
            h = Conv2D(filters, kernels, strides=1, padding="same")(h)
            h = LeakyReLU()(h)
        h = Conv2D(filters, kernels, strides=1, padding="same")(h)
        h = LeakyReLU()(h)
        h = MaxPooling2D(pool_size=(2,2))(h)
        h = Flatten()(h)
        if self.generator_model == "UAE":
            y_mean = Dense(self.latent_dim)(h)
            y = Lambda(self.uae_encoder_sampling)((y_mean, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y])
        elif self.generator_model == "AE":
            y = Dense(self.latent_dim)(h)
            encoder = keras.Model(inputs, y)
        else:
            #returns VAE or beta-VAE model
            y_mean = Dense(self.latent_dim)(h)
            y_sigma = Dense(self.latent_dim)(h)
            y = Lambda(self.vae_encoder_sampling)((y_mean, y_sigma, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y_sigma, y])
        # -----------------------Create decoder -------------------------------
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        h = Reshape((7, 7, 64))(h)
        h = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(h)
        h = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(h)
        h = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(h)
        decoder_outputs = Reshape((self.input_dim,))(h)
        decoder = keras.Model(latent_inputs, decoder_outputs)
        # -----------------instantiate generator model --------------------------------
        if self.generator_model == "UAE":
            outputs = decoder(encoder(inputs)[1])
            return keras.Model(inputs, outputs)
        elif self.generator_model == "AE":
            outputs = decoder(encoder(inputs))
            return keras.Model(inputs, outputs)
        else:
            #returns VAE or beta-VAE model
            outputs = decoder(encoder(inputs)[2])
            return keras.Model(inputs, outputs), encoder

    def a_u_model(self):
        # utility provider model
        inputs2 = keras.Input(shape=(self.input_dim,))
        converter = Reshape((28,28,1))(inputs2)
        h = Conv2D(32, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(converter)
        h = BatchNormalization()(h)
        h = Conv2D(32, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size = (2,2))(h)
        h = Dropout(0.3)(h)
        h = Conv2D(64, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = Conv2D(64, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size = (2,2))(h)
        h = Dropout(0.4)(h)
        h = Conv2D(128, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = Conv2D(128, kernel_size=3, activation='relu', kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size = (2,2))(h)
        h = Dropout(0.5)(h)
        h = Flatten()(h)
        h = Dense(128, activation = 'relu', kernel_initializer = "he_uniform")(h)
        h = BatchNormalization()(h)
        h = Dropout(0.6)(h)
        outputs2 = Dense(self.utility_output_dim, activation = 'softmax')(h)
        utility = keras.Model(inputs2, outputs2)
        return utility

    def a_p_model(self):
        #adversary model i.e. private inference model
        inputs1 = keras.Input(shape=(self.input_dim,))
        converter = Reshape((28,28,1))(inputs1)
        h = Conv2D(32, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(converter)
        h = BatchNormalization()(h)
        h = Conv2D(32, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size = (2,2))(h)
        h = Dropout(0.3)(h)
        h = Conv2D(64, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = Conv2D(64, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size = (2,2))(h)
        h = Dropout(0.4)(h)
        h = Conv2D(128, kernel_size=3, activation='relu',  kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = Conv2D(128, kernel_size=3, activation='relu', kernel_initializer = "he_uniform", padding = 'same')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size = (2,2))(h)
        h = Dropout(0.5)(h)
        h = Flatten()(h)
        h = Dense(128, activation = 'relu', kernel_initializer = "he_uniform")(h)
        h = BatchNormalization()(h)
        h = Dropout(0.6)(h)
        outputs1 = Dense(self.private_output_dim, activation = 'softmax')(h)
        private = keras.Model(inputs1, outputs1)
        return private


class UCIAdult(Sampling):
    def __init__(self, generator_model, inp=102, ap_out=2, au_out=2, latent_dim = 30):
        self.generator_model = generator_model
        self.input_dim = inp
        self.latent_dim = latent_dim
        self.private_output_dim = ap_out
        self.utility_output_dim = au_out

    def generator(self):
        #defines generator part of our model which consists of encoder and decoder
        # -------------------- Create encoder -----------------------------
        intermediate_dim = 128
        inputs = keras.Input(shape=(self.input_dim,))
        h = Dense(intermediate_dim, activation='relu')(inputs)
        h = Dense(intermediate_dim, activation='relu')(h)
        h = Dense(intermediate_dim, activation='relu')(h)
        if self.generator_model == "UAE":
            y_mean = Dense(self.latent_dim)(h)
            y = Lambda(self.uae_encoder_sampling)((y_mean, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y])
        elif self.generator_model == "AE":
            y = Dense(self.latent_dim)(h)
            encoder = keras.Model(inputs, y)
        else:
            #returns VAE or beta-VAE model
            y_mean = Dense(self.latent_dim)(h)
            y_sigma = Dense(self.latent_dim)(h)
            y = Lambda(self.vae_encoder_sampling)((y_mean, y_sigma, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y_sigma, y])
        # -----------------------Create decoder -------------------------------
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h = Dense(intermediate_dim, activation='relu')(latent_inputs)
        h = Dense(intermediate_dim, activation='relu')(h)
        x = Dense(self.input_dim, activation='linear')(h)
        decoder = keras.Model(latent_inputs, x)
        # -----------------instantiate generator model --------------------------------
        if self.generator_model == "UAE":
            outputs = decoder(encoder(inputs)[1])
            return keras.Model(inputs, outputs)
        elif self.generator_model == "AE":
            outputs = decoder(encoder(inputs))
            return keras.Model(inputs, outputs)
        else:
            #returns VAE or beta-VAE model
            outputs = decoder(encoder(inputs)[2])
            return keras.Model(inputs, outputs), encoder

    def a_u_model(self):
        #utility provider model
        input1 = keras.Input(shape=(self.input_dim,))
        h = Dense(256, activation='relu')(input1)
        h = Dropout(0.2)(h)
        h = Dense(256, activation='relu')(h)
        h = Dropout(0.3)(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(0.4)(h)
        output1 = Dense(self.utility_output_dim, activation='softmax')(h)
        utility = keras.Model(inputs = input1, outputs = output1)
        return utility

    def a_p_model(self):
        # adversary model i.e. private inference model
        input1 = keras.Input(shape=(self.input_dim,))
        h = Dense(256, activation='relu')(input1)
        h = Dropout(0.2)(h)
        h = Dense(256, activation='relu')(h)
        h = Dropout(0.3)(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(0.4)(h)
        output1 = Dense(self.private_output_dim, activation='softmax')(h)
        private = keras.Model(inputs = input1, outputs = output1)
        return private

class USCensus(Sampling):
    def __init__(self, generator_model, inp=14, ap_out=2, au_out=2, latent_dim = 12):
        self.generator_model = generator_model
        self.input_dim = inp
        self.latent_dim = latent_dim
        self.private_output_dim = ap_out
        self.utility_output_dim = au_out

    def generator(self):
        #defines generator part of our model which consists of encoder and decoder
        intermediate_dim = 128
        # -------------------- Create encoder -----------------------------
        inputs = keras.Input(shape=(self.input_dim,))
        h = Dense(intermediate_dim, activation='relu')(inputs)
        h = Dense(intermediate_dim, activation='relu')(h)
        h = Dense(intermediate_dim, activation='relu')(h)
        if self.generator_model == "UAE":
            y_mean = Dense(self.latent_dim)(h)
            y = Lambda(self.uae_encoder_sampling)((y_mean, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y])
        elif self.generator_model == "AE":
            y = Dense(self.latent_dim)(h)
            encoder = keras.Model(inputs, y)
        else:
            #returns VAE or beta-VAE model
            y_mean = Dense(self.latent_dim)(h)
            y_sigma = Dense(self.latent_dim)(h)
            y = Lambda(self.vae_encoder_sampling)((y_mean, y_sigma, self.latent_dim))
            encoder = keras.Model(inputs, [y_mean, y_sigma, y])
        # -----------------------Create decoder -------------------------------
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h = Dense(intermediate_dim, activation='relu')(latent_inputs)
        h = Dense(intermediate_dim, activation='relu')(h)
        x = Dense(self.input_dim, activation='linear')(h)
        decoder = keras.Model(latent_inputs, x)
        # -----------------instantiate generator model --------------------------------
        if self.generator_model == "UAE":
            outputs = decoder(encoder(inputs)[1])
            return keras.Model(inputs, outputs)
        elif self.generator_model == "AE":
            outputs = decoder(encoder(inputs))
            return keras.Model(inputs, outputs)
        else:
            #returns VAE or beta-VAE model
            outputs = decoder(encoder(inputs)[2])
            return keras.Model(inputs, outputs), encoder

    def a_u_model(self):
        # utility provider model
        input1 = keras.Input(shape=(self.input_dim,))
        hidden1 = Dense(64, activation='relu')(input1)
        hidden2 = Dense(64, activation='relu')(hidden1)
        hidden3 = Dense(64, activation='relu')(hidden2)
        output1 = Dense(self.utility_output_dim, activation='softmax')(hidden3)
        utility = keras.Model(inputs = input1, outputs = output1)
        return utility

    def a_p_model(self):
        input2 = keras.Input(shape=(self.input_dim,))
        hidden1 = Dense(64, activation='relu')(input2)
        hidden2 = Dense(64, activation='relu')(hidden1)
        hidden3 = Dense(64, activation='relu')(hidden2)
        output2 = Dense(self.private_output_dim, activation='softmax')(hidden3)
        private = keras.Model(inputs = input2, outputs = output2)
        return private
