# Import all the required libraries and packages
import os
import argparse
import keras
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from keras.layers import Dense, Dropout, Lambda
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as utils
from tensorflow.keras.optimizers import SGD
from keras import backend as K
import collections
from model import Sampling
from loss_curves import Loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# Command line arguments
parser = argparse.ArgumentParser(description='Implement Data type aware - PUPET')
parser.add_argument('-g', '--generator', type=str, metavar='', required=True, help = 'Generator name: UAE or AE or VAE.')
parser.add_argument('-e', '--epochs', type=int, metavar='', default = 100, help = 'Default epochs: 100')
parser.add_argument('-p', '--lambda_p', type = int, metavar='', default = 10, help = "Lambda_P value. Suggested Range -> (0,10)")
args = parser.parse_args()

class Dataset:
    def get_data(self, main_dir):
        '''
        Load dataset and perform preprocessing
        '''
        df = pd.read_csv(os.path.join(main_dir,'adult.csv'))
        df = df.replace({'?':np.nan})
        df = df.dropna()
        df1 = pd.get_dummies(df)
        train, test = train_test_split(df1, test_size = 0.2, random_state = 42)
        income_train = np.array(train[['income_<=50K','income_>50K']])
        utility_test_true_labels = np.array(test[['income_<=50K','income_>50K']])
        gender_train = np.array(train[['gender_Male', 'gender_Female']])
        private_test_true_labels = np.array(test[['gender_Male', 'gender_Female']])
        x_train = (train.drop(['income_<=50K','income_>50K','gender_Male', 'gender_Female'],axis='columns'))
        x_test = (test.drop(['income_<=50K','income_>50K','gender_Male', 'gender_Female'],axis='columns'))
        standard_scaler1 = preprocessing.MinMaxScaler(feature_range=(0,1))
        standard_scaler1.fit(x_train)
        x_train = standard_scaler1.transform(x_train)
        x_test = standard_scaler1.transform(x_test)
        return x_train, x_test, gender_train, private_test_true_labels, income_train, utility_test_true_labels



class UCIAdult(Sampling):
    def __init__(self, generator_model, inp=102, ap_out=2, au_out=2, latent_dim = 30):
        self.generator_model = generator_model
        self.input_dim = inp
        self.latent_dim = latent_dim
        self.private_output_dim = ap_out
        self.utility_output_dim = au_out

    @staticmethod
    def uae_encoder_sampling(args):
        #re-parametrization technique
        y_mean = args[0]
        sigma = 0.1
        latent_dim = args[1]
        epsilon = K.random_normal(shape=(K.shape(y_mean)[0], latent_dim),
                                  mean=0, stddev = 1)
        z = epsilon * sigma + y_mean
        #z = tf.squeeze(z, axis = 0)
        return z


    def sampling_step(self, inputs, h, dimension):
        if self.generator_model == "UAE":
            y_mean = Dense(dimension)(h)
            y = Lambda(self.uae_encoder_sampling)((y_mean, dimension))
            model = keras.Model(inputs, [y_mean, y])
        elif self.generator_model == "AE":
            y = Dense(dimension)(h)
            model = keras.Model(inputs, y)
        else:
            #returns VAE model incase if the model is not UAE or VAE
            y_mean = Dense(dimension)(h)
            y_sigma = Dense(dimension)(h)
            y = Lambda(self.vae_encoder_sampling)((y_mean, y_sigma, dimension))
            model = keras.Model(inputs, [y_mean, y_sigma, y])
        return model


    def generator(self):
        intermediate_dim = 128
        # -------------------- Create encoder -----------------------------
        inputs = keras.Input(shape=(self.input_dim,), name = "encoder input")
        h = Dense(intermediate_dim, activation='relu')(inputs)
        h = Dense(intermediate_dim, activation='relu')(h)
        h = Dense(intermediate_dim, activation='relu')(h)
        encoder = self.sampling_step(inputs, h, self.latent_dim)

        # -----------------------Create decoder -------------------------------
        latent_inputs = keras.Input(shape=(self.latent_dim,), name='decoder input')
        h = Dense(intermediate_dim, activation='relu')(latent_inputs)
        h = Dense(intermediate_dim, activation='relu')(h)
        h = Dense(self.input_dim, activation='relu')(h)
        decoder = keras.Model(latent_inputs, h)

        # -----------------instantiate generator model --------------------------------
        if self.generator_model == "UAE":
            outputs = decoder(encoder(inputs)[1])
            return keras.Model(inputs, outputs)
        elif self.generator_model == "AE":
            outputs = decoder(encoder(inputs))
            return keras.Model(inputs, outputs)
        else:
            #return VAE
            outputs = decoder(encoder(inputs)[2])
            return keras.Model(inputs, outputs), encoder

    # ------------Utility Model----------------------
    def a_u_model(self):
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

    # ------------Adversary (Private) Model----------------------
    def a_p_model(self):
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

class TrainPreprocessing:
    def __init__(self, generator):
        self.generator = generator

    def optimizers(self):
        generator_optimizer1 = tf.keras.optimizers.Adam()
        private_optimizer1 = tf.keras.optimizers.Adam()
        utility_optimizer1 = tf.keras.optimizers.Adam()
        return generator_optimizer1, private_optimizer1, utility_optimizer1

    def losses(self):
        if self.generator == "VAE":
            gen_loss1 = keras.losses.BinaryCrossentropy()
        else:
            gen_loss1 = keras.losses.MeanSquaredError()
        private_loss1 = keras.losses.CategoricalCrossentropy()
        utility_loss1 = keras.losses.CategoricalCrossentropy()
        return gen_loss1, private_loss1, utility_loss1


    # Make batches of data
    def datasets(self, x_train, private_train, utility_train):
        batch_size = 512
        train_dataset1 = tf.data.Dataset.from_tensor_slices((x_train, private_train, utility_train))
        train_dataset1 = train_dataset1.shuffle(buffer_size = 1024).batch(batch_size)
        return train_dataset1


@tf.function
def train_step(train_dataset, test_dataset, lambda_p, lambda_u, generator_type):
    '''
    define custom training loop for data type aware conditions
    '''
    x_train1, pri_train1, uti_train1 = train_dataset
    x_test1, pri_test1, uti_test1 = test_dataset
    #Train generator model: UAE or AE or VAE
    with tf.GradientTape() as tape:
        data = generator_model(x_train1)
        private_predicted = a_p(data)
        p1_loss = private_loss(pri_train1, private_predicted)
        utility_predicted = a_u(data)
        u1_loss = utility_loss(uti_train1, utility_predicted)
        initial_loss = gen_loss_1(x_train1, data) * 102

        if generator_type == "VAE":
            mean = encoder(x_train1)[0]
            log_sigma = encoder(x_train1)[1]
            kl_loss = 1 + log_sigma - K.square(mean) - K.exp(log_sigma)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            initial_loss =  K.mean(initial_loss +  kl_loss)
        final_loss = (initial_loss) + (lambda_u * u1_loss) - (lambda_p * p1_loss)

    # For validation or test dataset
    data_test = generator_model(x_test1)
    private_predicted_test = a_p(data_test)
    p1_loss_test = private_loss(pri_test1, private_predicted_test)
    utility_predicted_test = a_u(data_test)
    u1_loss_test = utility_loss(uti_test1, utility_predicted_test)
    initial_loss_test = gen_loss_1(x_test1, data_test) * 102


    if generator_type == "VAE":
        mean = encoder(x_test1)[0]
        log_sigma = encoder(x_test1)[1]
        kl_loss = 1 + log_sigma - K.square(mean) - K.exp(log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        initial_loss_test =  K.mean(initial_loss_test +  kl_loss)
    final_loss_test = (initial_loss_test)  + (lambda_u * u1_loss_test) - (lambda_p * p1_loss_test)


    #back propagation and updating weights
    generator_grads = tape.gradient(final_loss, generator_model.trainable_weights)
    generator_optimizer.apply_gradients(zip(generator_grads, generator_model.trainable_weights))

    #Train Private Classifier
    with tf.GradientTape() as tape:
        data = generator_model(x_train1)
        private_predicted = a_p(data)
        p_loss = private_loss(pri_train1, private_predicted)

    # For validation or test data
    data_test = generator_model(x_test1)
    private_predicted_test = a_p(data_test)
    p_loss_test = private_loss(pri_test1, private_predicted_test)

    #back propagation and updating weights
    private_grads = tape.gradient(p_loss, a_p.trainable_weights)
    private_optimizer.apply_gradients(zip(private_grads, a_p.trainable_weights))

    #Train Utility Classifier
    with tf.GradientTape() as tape:
        data = generator_model(x_train1)
        utility_predicted = a_u(data)
        u_loss = utility_loss(uti_train1, utility_predicted)

    # For validation or test data
    data_test = generator_model(x_test1)
    utility_predicted_test = a_u(data_test)
    u_loss_test = utility_loss(uti_test1, utility_predicted_test)

    #back propagation and updating weights
    utility_grads = tape.gradient(u_loss, a_u.trainable_weights)
    utility_optimizer.apply_gradients(zip(utility_grads, a_u.trainable_weights))
    return final_loss, p_loss, u_loss, final_loss_test, p_loss_test, u_loss_test



if __name__ == "__main__":
    main_dir = "uae-pupet"
    dataset = Dataset()
    x_train, x_test, private_train_true_labels, private_test_true_labels, utility_train_true_labels, utility_test_true_labels = dataset.get_data(main_dir)
    models = UCIAdult(args.generator)
    if args.generator == "VAE":
        generator_model, encoder = models.generator()
    else:
        generator_model = models.generator()
    a_p = models.a_p_model()
    a_u = models.a_u_model()
    preprocess = TrainPreprocessing(args.generator)
    generator_optimizer, private_optimizer, utility_optimizer = preprocess.optimizers()
    gen_loss_1, private_loss, utility_loss = preprocess.losses()
    train_dataset = preprocess.datasets(x_train, private_train_true_labels, utility_train_true_labels)
    test_dataset = (x_test, private_test_true_labels, utility_test_true_labels)
    #BEST PUPET hyperparameters after multiple tests.
    lambda_u = 1
    loss_generator, loss_adversary, loss_utility = [], [], []
    loss_generator_test, loss_adversary_test, loss_utility_test = [], [], []
    for epoch in range(args.epochs):
        #print("Epoch: {}".format(epoch))
        for step, dataset in enumerate(train_dataset):
            gen_loss1, pri_loss1, uti_loss1, gen_loss1_test, pri_loss1_test, uti_loss1_test = train_step(dataset, test_dataset, args.lambda_p, lambda_u, args.generator)
            loss_generator.append(gen_loss1)
            loss_adversary.append(pri_loss1)
            loss_utility.append(uti_loss1)

            loss_generator_test.append(gen_loss1_test)
            loss_adversary_test.append(pri_loss1_test)
            loss_utility_test.append(uti_loss1_test)
    #draw loss LossCurves
    curves = Loss()
    curves.curves(main_dir, args.lambda_p, "Data-Type-Aware", args.generator, loss_generator, loss_adversary, loss_utility, loss_generator_test, loss_adversary_test, loss_utility_test)
    # Get results after training
    private_models = []
    utility_models = []
    # test under weak adversary and utility provider
    private_models.append(keras.models.load_model(os.path.join(main_dir, "adversaries_and_utility","UCIAdult",'weak-adversary')))
    utility_models.append(keras.models.load_model(os.path.join(main_dir, "adversaries_and_utility","UCIAdult",'weak-utility')))

    # adversary and utility which was dynamically trained during the recent optimization
    private_models.append(a_p)
    utility_models.append(a_u)
    # Generate Privatized Data
    private_data = generator_model.predict([x_test])


    # Calculate the level of noise for continuous features: Defined by MSE
    cont_noise = keras.losses.MeanSquaredError()
    cont1 = private_data[:,:6]
    noise = cont_noise(x_test[:,:6], cont1)

    # Calculate the level of noise for categorical features: Defined by number of category flips
    cat1 = private_data[:,6:13]
    cat1_dup = np.zeros_like(cat1)
    cat1_dup[np.arange(len(cat1)), cat1.argmax(1)] = 1

    old = np.argmax(x_test[:,6:13], axis = 1)
    new = np.argmax(cat1_dup, axis = 1)

    labels_change_count = []

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)

    cat2 = private_data[:,13:29]
    cat2_dup = np.zeros_like(cat2)
    cat2_dup[np.arange(len(cat2)), cat2.argmax(1)] = 1

    old = np.argmax(x_test[:,13:29],axis = 1)
    new = np.argmax(cat2_dup, axis = 1)

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)


    cat3 = private_data[:,29:36]
    cat3_dup = np.zeros_like(cat3)
    cat3_dup[np.arange(len(cat3)), cat3.argmax(1)] = 1

    old = np.argmax(x_test[:,29:36], axis = 1)
    new = np.argmax(cat3_dup, axis = 1)

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)

    cat4 = private_data[:,36:50]
    cat4_dup = np.zeros_like(cat4)
    cat4_dup[np.arange(len(cat4)), cat4.argmax(1)] = 1

    old = np.argmax(x_test[:,36:50], axis = 1)
    new = np.argmax(cat4_dup, axis = 1)

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)

    cat5 = private_data[:,50:56]
    cat5_dup = np.zeros_like(cat5)
    cat5_dup[np.arange(len(cat5)), cat5.argmax(1)] = 1

    old = np.argmax(x_test[:,50:56], axis = 1)
    new = np.argmax(cat5_dup, axis = 1)

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)

    cat6 = private_data[:,56:61]
    cat6_dup = np.zeros_like(cat6)
    cat6_dup[np.arange(len(cat6)), cat6.argmax(1)] = 1

    old = np.argmax(x_test[:,56:61], axis = 1)
    new = np.argmax(cat6_dup, axis = 1)

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)

    cat7 = private_data[:,61:]
    cat7_dup = np.zeros_like(cat7)
    cat7_dup[np.arange(len(cat7)), cat7.argmax(1)] = 1

    old = np.argmax(x_test[:,61:], axis = 1)
    new = np.argmax(cat7_dup, axis = 1)

    count = 0
    for i, j in zip(old, new):
        if i!=j:
            count += 1
    labels_change_count.append(count)

    # Cocatenated Data Type aware Private Data
    new_private_data = np.concatenate([cont1, cat1_dup, cat2_dup, cat3_dup, cat4_dup, cat5_dup, cat6_dup, cat7_dup], axis = 1)


    # Make strong adversary models
    strong_adversary = models.a_p_model()
    strong_utility = models.a_u_model()

    private_train_data = generator_model.predict([x_train])
    epoch= 40

    # Train strong adversary
    #print("Private Training")
    strong_adversary.compile(optimizer = 'SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    strong_adversary.fit(private_train_data, private_train_true_labels, validation_data = (new_private_data, private_test_true_labels), batch_size= 512, epochs=epoch, shuffle = True, verbose = 0)

    #print("Utility Training")
    strong_utility.compile(optimizer = 'SGD',loss='categorical_crossentropy',metrics=['accuracy'])
    strong_utility.fit(private_train_data, utility_train_true_labels, validation_data = (new_private_data, utility_test_true_labels), batch_size= 512, epochs=epoch, shuffle = True, verbose = 0)

    private_models.append(strong_adversary)
    utility_models.append(strong_utility)


    # Private result
    private_acc, private_auroc = [], []
    for classifier in private_models:
        predicted = np.argmax(classifier.predict(new_private_data), axis = 1)
        actual = np.argmax(private_test_true_labels, axis = 1)
        private_acc.append(accuracy_score(actual, predicted))
        private_auroc.append(roc_auc_score(actual, predicted, average = 'macro'))

    # Utilities result
    utility_acc, utility_auroc = [], []
    for classifier in utility_models:
        predicted = np.argmax(classifier.predict(new_private_data), axis = 1)
        actual = np.argmax(utility_test_true_labels, axis = 1)
        utility_acc.append(accuracy_score(actual, predicted))
        utility_auroc.append(roc_auc_score(actual, predicted, average = 'macro'))


    #Result of best performing adversary based on accuracy
    max_private_acc = max(private_acc)
    index_p = private_acc.index(max_private_acc)
    max_private_auroc = private_auroc[index_p]
    #Result of best performing utility provider based on accuracy
    max_utility_acc = max(utility_acc)
    index_u = utility_acc.index(max_utility_acc)
    max_utility_auroc = utility_auroc[index_u]
    # Save results in dataset respective folder as a text file.
    with open(os.path.join(main_dir,"results","Data-Type-Aware",str(args.generator)+"-"+str(args.lambda_p)+"-private_acc.txt"), "a+") as file:
        file.write(str(max_private_acc)+"\n")
    with open(os.path.join(main_dir,"results","Data-Type-Aware",str(args.generator)+"-"+str(args.lambda_p)+"-utility_acc.txt"), "a+") as file:
        file.write(str(max_utility_acc)+"\n")
    with open(os.path.join(main_dir,"results","Data-Type-Aware",str(args.generator)+"-"+str(args.lambda_p)+"-private_auroc.txt"), "a+") as file:
        file.write(str(max_private_auroc)+"\n")
    with open(os.path.join(main_dir,"results","Data-Type-Aware",str(args.generator)+"-"+str(args.lambda_p)+"-utility_auroc.txt"), "a+") as file:
        file.write(str(max_utility_auroc)+"\n")

    with open(os.path.join(main_dir,"results","Data-Type-Aware",str(args.generator)+"-"+str(args.lambda_p)+"-mse_noise"), "a+") as file:
        file.write(str(noise)+"\n")

    for cat, noise in enumerate(labels_change_count):
        with open(os.path.join(main_dir,"results","Data-Type-Aware",str(args.generator)+"-"+str(args.lambda_p)+"-cat_noise"+str(cat+1)+".txt"), "a+") as file:
            file.write(str(noise)+"\n")


    os.kill(os.getpid(), 9)
