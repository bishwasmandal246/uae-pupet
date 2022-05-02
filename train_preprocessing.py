import sys
import model
import keras
import tensorflow as tf
from dataset import Dataset

class TrainPreprocessing:
    def __init__(self, main_dir, dataset, generator):
        self.main_dir = main_dir
        self.dataset = dataset
        self.generator = generator

    def get_data(self):
        retrieve_data = Dataset(self.main_dir)
        if self.dataset == "MNIST":
            return retrieve_data.mnist_data()
        elif self.dataset == "FashionMNIST":
             return retrieve_data.fashion_mnist_data()
        elif self.dataset == "UCIAdult":
            return retrieve_data.uci_adult()
        elif self.dataset == "USCensus":
            return retrieve_data.us_census_demographic_data()
        else:
            print("Incorrect Dataset input given. Refer to list of supported datasets and try again...")
            print("Program terminated!")
            sys.exit(0)

    def get_model(self):
        if self.dataset == "MNIST":
            return model.MNIST(self.generator)
        elif self.dataset == "FashionMNIST":
             return model.FashionMNIST(self.generator)
        elif self.dataset == "UCIAdult":
            return model.UCIAdult(self.generator)
        else:
            return model.USCensus(self.generator)

    def optimizers(self):
        generator_optimizer1 = tf.keras.optimizers.Adam()
        private_optimizer1 = tf.keras.optimizers.Adam()
        utility_optimizer1 = tf.keras.optimizers.Adam()
        return generator_optimizer1, private_optimizer1, utility_optimizer1

    def losses(self):
        if self.generator == "VAE" or self.generator == "b-VAE":
            #Change as per the dataset and requirements
            generator_loss1 = keras.losses.BinaryCrossentropy()
        else:
            generator_loss1 = keras.losses.MeanSquaredError()
        private_loss1 = keras.losses.CategoricalCrossentropy()
        utility_loss1 = keras.losses.CategoricalCrossentropy()
        return generator_loss1, private_loss1, utility_loss1

    def datasets(self, x_train, pri_train, uti_train):
        if self.dataset == "USCensus":
            batch_size = 512
        else:
            batch_size = 256
        train_dataset1 = tf.data.Dataset.from_tensor_slices((x_train, pri_train, uti_train))
        train_dataset1 = train_dataset1.shuffle(buffer_size = 1024).batch(batch_size)
        return train_dataset1
