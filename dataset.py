import os
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow.keras.utils as utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, dir):
        self.dir = dir

    def mnist_data(self):
        '''
        Make two variables to denote private and utility features based on the problem setting. Here private features
        refer to whether a digit is odd or even and utility refers to a number is >=5 or not.
        '''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # normalizing pixel values to the range 0 to 1
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        # changing (28,28) into one dimensional data of size 784
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        private_train = []
        utility_train = []
        for i in y_train:
            if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
                private_train.append(0)
            else:
                private_train.append(1)
            if i < 5:
                utility_train.append(0)
            else:
                utility_train.append(1)
        private_train_true_labels = utils.to_categorical(private_train)
        utility_train_true_labels = utils.to_categorical(utility_train)
        private_test = []
        utility_test = []
        for i in y_test:
            if i == 0 or i == 2 or i == 4 or i == 6 or i == 8:
                private_test.append(0)
            else:
                private_test.append(1)
            if i < 5:
                utility_test.append(0)
            else:
                utility_test.append(1)
        private_test_true_labels = utils.to_categorical(private_test)
        utility_test_true_labels = utils.to_categorical(utility_test)
        return x_train, x_test, private_train_true_labels, private_test_true_labels, utility_train_true_labels, utility_test_true_labels

    def fashion_mnist_data(self):
        '''
        Make two variables to denote private and utility features based on the problem setting. Here private features
        refer to clothing attire identities and utility features refer to whether an attrire is upper body clothing attire or
        miscellaneous (anything except first utility label)
        '''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # normalizing pixel values to the range 0 to 1
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        # changing (28,28) into one dimensional data of size 784
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        private_train_true_labels = utils.to_categorical(y_train)
        private_test_true_labels = utils.to_categorical(y_test)
        utility_train = []
        utility_test = []
        for i in y_train:
          if i == 0 or i == 2 or i== 3 or i ==4 or i==6:
            utility_train.append(0)
          else:
            utility_train.append(1)
        utility_train_true_labels = utils.to_categorical(utility_train)
        for i in y_test:
          if i == 0 or i == 2 or i== 3 or i ==4 or i==6:
            utility_test.append(0)
          else:
            utility_test.append(1)
        utility_test_true_labels = utils.to_categorical(utility_test)
        return x_train, x_test, private_train_true_labels, private_test_true_labels, utility_train_true_labels, utility_test_true_labels

    def uci_adult(self):
        '''
        Load uci adult dataset and perform preprocessing.
        Gender is private feature and income is utility feature
        '''
        df = pd.read_csv(os.path.join(self.dir,'adult.csv'))
        df = df.replace({'?':np.nan})
        df = df.dropna()
        df1 = pd.get_dummies(df)
        train, test = train_test_split(df1, test_size = 0.2, random_state = 42)
        utility_train_true_labels = np.array(train[['income_<=50K','income_>50K']])
        utility_test_true_labels = np.array(test[['income_<=50K','income_>50K']])
        private_train_true_labels = np.array(train[['gender_Male', 'gender_Female']])
        private_test_true_labels = np.array(test[['gender_Male', 'gender_Female']])
        x_train = (train.drop(['income_<=50K','income_>50K','gender_Male', 'gender_Female'],axis='columns'))
        x_test = (test.drop(['income_<=50K','income_>50K','gender_Male', 'gender_Female'],axis='columns'))
        standard_scaler = preprocessing.StandardScaler()
        standard_scaler.fit(x_train)
        x_train = standard_scaler.transform(x_train)
        x_test = standard_scaler.transform(x_test)
        return x_train, x_test, private_train_true_labels, private_test_true_labels, utility_train_true_labels, utility_test_true_labels

    def us_census_demographic_data(self):
        '''
        Load dataset, preprocess and turn regression problem into a classification problem.
        Income is private feature and Employed is utility feature.
        '''
        df1 = pd.read_csv(os.path.join(self.dir,'TrainData.csv'))
        df2 = pd.read_csv(os.path.join(self.dir,'TestData.csv'))
        #Classification labels
        for i in range(len(df1)):
            if df1.Employed[i] <= 2000:
                df1.Employed[i] = 0
            else:
                df1.Employed[i] = 1
            if df1.Income[i] <= 55000:
                df1.Income[i] = 0
            else:
                df1.Income[i] = 1
        for i in range(len(df2)):
            if df2.Employed[i] <= 2000:
                df2.Employed[i] = 0
            else:
                df2.Employed[i] = 1
            if df2.Income[i] <= 55000:
                df2.Income[i] = 0
            else:
                df2.Income[i] = 1
        private_train_true_labels = utils.to_categorical(df1.Income)
        utility_train_true_labels = utils.to_categorical(df1.Employed)
        private_test_true_labels = utils.to_categorical(df2.Income)
        utility_test_true_labels = utils.to_categorical(df2.Employed)
        df1 = df1.drop(['Income','Employed'],axis='columns')
        df2 = df2.drop(['Income','Employed'],axis='columns')
        standard_scaler1 = preprocessing.StandardScaler()
        x_train = standard_scaler1.fit_transform(df1)
        standard_scaler2 = preprocessing.StandardScaler()
        x_test = standard_scaler2.fit_transform(df2)
        return x_train, x_test, private_train_true_labels, private_test_true_labels, utility_train_true_labels, utility_test_true_labels
