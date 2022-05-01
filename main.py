# Import all the required libraries, packages and files
import os
import sys
import argparse
from dataset import Dataset
import model
import original_vs_private_image
import keras
import tensorflow as tf
import numpy as np
from keras import backend as K
from numpy import random
import pandas as pd
import tensorflow.keras.utils as utils
from get_results import Results
from loss_curves import Loss
from train_preprocessing import TrainPreprocessing
from tensorflow.keras.layers import Dense, Reshape, Lambda, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU, BatchNormalization, Conv2DTranspose

# Command line arguments
parser = argparse.ArgumentParser(description='Implement PUPET')
parser.add_argument('-d', '--dataset', type=str, metavar='', required=True, help = 'Dataset name: MNIST, FashionMNIST, UCIAdult or USCensus.')
parser.add_argument('-g', '--generator', type=str, metavar='', required=True, help = 'Generator name: UAE, or AE, or VAE, or b-VAE.')
parser.add_argument('-e', '--epochs', type=int, metavar='', default = 40, help = 'Default epochs: 40')
parser.add_argument('-p', '--lambda_p', type = int, metavar='', required = True, help = "Favorable range: 0 to 100. Value only used when overwrite is true.")
parser.add_argument('-o', '--lambda_p_overwrite', type = str, metavar='', required = True, help = 'Accepts "true" or "false". Lambda_p value mentioned above only effective when overwrite = true.')
args = parser.parse_args()

# Custom training loop
@tf.function
def train_step(train_dataset, test_dataset, lambda_p, lambda_u, original_dim, generator_type):
    '''
    define custom training loop
    '''
    x_train1, pri_train1, uti_train1 = train_dataset
    x_test1, pri_test1, uti_test1 = test_dataset
    #Train Generator Model: UAE, AE or VAE
    with tf.GradientTape() as tape:
        private_predicted = a_p(generator_model(x_train1))
        p1_loss = private_loss(pri_train1, private_predicted)
        utility_predicted = a_u(generator_model(x_train1))
        u1_loss = utility_loss(uti_train1, utility_predicted)
        generator_predicted = generator_model(x_train1)
        initial_loss = generator_loss(x_train1, generator_predicted)
        initial_loss *= original_dim
        if generator_type == "VAE" or generator_type == "b-VAE":
            mean = encoder(x_train1)[0]
            log_sigma = encoder(x_train1)[1]
            kl_loss = 1 + log_sigma - K.square(mean) - K.exp(log_sigma)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            if generator_type == "VAE":
                beta = 1
            elif generator_type == "b-VAE":
                beta = 2
            initial_loss =  K.mean(initial_loss + (beta * kl_loss))
        final_loss = initial_loss + (lambda_u * u1_loss) - (lambda_p * p1_loss)

    # For validation or test data
    private_predicted_test = a_p(generator_model(x_test1))
    p1_loss_test = private_loss(pri_test1, private_predicted_test)
    utility_predicted_test = a_u(generator_model(x_test1))
    u1_loss_test = utility_loss(uti_test1, utility_predicted_test)
    generator_predicted_test = generator_model(x_test1)
    initial_loss_test = generator_loss(x_test1, generator_predicted_test)
    initial_loss_test *= original_dim
    if generator_type == "VAE" or generator_type == "b-VAE":
        mean_test = encoder(x_test1)[0]
        log_sigma_test = encoder(x_test1)[1]
        kl_loss_test = 1 + log_sigma_test - K.square(mean_test) - K.exp(log_sigma_test)
        kl_loss_test = K.sum(kl_loss_test, axis=-1)
        kl_loss_test *= -0.5
        if generator_type == "VAE":
            beta = 1
        elif generator_type == "b-VAE":
            beta = 2
        initial_loss_test =  K.mean(initial_loss_test + (beta * kl_loss_test))
    final_loss_test = initial_loss_test + (lambda_u * u1_loss_test) - (lambda_p * p1_loss_test)

    # back propagation and update generator parameters
    generator_grads = tape.gradient(final_loss, generator_model.trainable_weights)
    generator_optimizer.apply_gradients(zip(generator_grads, generator_model.trainable_weights))

    #Train Private Classifier
    with tf.GradientTape() as tape:
        private_predicted = a_p(generator_model(x_train1))
        p_loss = private_loss(pri_train1, private_predicted)

    # For validation or test data
    private_predicted_test = a_p(generator_model(x_test1))
    p_loss_test = private_loss(pri_test1, private_predicted_test)

    # back propagation and update adversary parameters
    private_grads = tape.gradient(p_loss, a_p.trainable_weights)
    private_optimizer.apply_gradients(zip(private_grads, a_p.trainable_weights))

    #Train Utility Classifier
    with tf.GradientTape() as tape:
        utility_predicted = a_u(generator_model(x_train1))
        u_loss = utility_loss(uti_train1, utility_predicted)

    # For validation or test data
    utility_predicted_test = a_u(generator_model(x_test1))
    u_loss_test = utility_loss(uti_test1, utility_predicted_test)

    # back propagation and update utility provider parameters
    utility_grads = tape.gradient(u_loss, a_u.trainable_weights)
    utility_optimizer.apply_gradients(zip(utility_grads, a_u.trainable_weights))

    return final_loss, p_loss, u_loss, final_loss_test, p_loss_test, u_loss_test

# Main: Train and get results
if __name__ == '__main__':
    main_dir = "uae-pupet"
    preprocess = TrainPreprocessing(main_dir, args.dataset, args.generator)
    x_train, x_test, private_train_true_labels, private_test_true_labels, utility_train_true_labels, utility_test_true_labels = preprocess.get_data()
    models = preprocess.get_model()
    if args.generator == "VAE" or args.generator == "b-VAE":
        generator_model, encoder = models.generator()
    else:
        generator_model = models.generator()
    a_p = models.a_p_model()
    a_u = models.a_u_model()
    generator_optimizer, private_optimizer, utility_optimizer = preprocess.optimizers()
    generator_loss, private_loss, utility_loss = preprocess.losses()
    train_dataset = preprocess.datasets(x_train, private_train_true_labels, utility_train_true_labels)
    test_dataset = (x_test, private_test_true_labels, utility_test_true_labels)
    #Best working UAE-PUPET hyperparameters after multiple tests.
    lambda_u = 1
    if args.dataset == "MNIST":
        if args.lambda_p_overwrite == "false":
            lambda_p = 40
        else:
            lambda_p = args.lambda_p
    elif args.dataset == "FashionMNIST":
        if args.lambda_p_overwrite == "false":
            lambda_p = 90
        else:
            lambda_p = args.lambda_p
    elif args.dataset == "UCIAdult":
        if args.lambda_p_overwrite == "false":
            lambda_p = 100
        else:
            lambda_p = args.lambda_p
    else:
        if args.lambda_p_overwrite == "false": #US Census dataset
            lambda_p = 70
        else:
            lambda_p = args.lambda_p
    #Train PUPET (a privacy utility preserving end-to-end Transformation)
    original_dim = x_train.shape[1]
    loss_generator, loss_adversary, loss_utility = [], [], []
    loss_generator_test, loss_adversary_test, loss_utility_test = [], [], []
    for epoch in range(args.epochs):
        for step, dataset in enumerate(train_dataset):
            gen_loss1, pri_loss1, uti_loss1, gen_loss1_test, pri_loss1_test, uti_loss1_test = train_step(dataset, test_dataset, lambda_p, lambda_u, original_dim, args.generator)
            loss_generator.append(gen_loss1)
            loss_adversary.append(pri_loss1)
            loss_utility.append(uti_loss1)

            loss_generator_test.append(gen_loss1_test)
            loss_adversary_test.append(pri_loss1_test)
            loss_utility_test.append(uti_loss1_test)
    #draw loss LossCurves
    curves = Loss()
    curves.curves(main_dir, lambda_p, args.dataset, args.generator, loss_generator, loss_adversary, loss_utility, loss_generator_test, loss_adversary_test, loss_utility_test)
    # Get results after training
    result = Results(main_dir, generator_model, args.dataset, a_p, a_u, args.generator)
    result.get_results(x_test, private_test_true_labels, utility_test_true_labels, lambda_p)
    os.kill(os.getpid(), 9)
