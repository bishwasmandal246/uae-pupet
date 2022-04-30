import os
import keras
import numpy as np
import tensorflow.keras.utils as utils
from original_vs_private_image import Draw
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

class Results:
    def __init__(self, main_dir, generator, dataset, a_p, a_u, gen_name):
        self.main_dir = main_dir
        self.generator = generator
        self.dataset = dataset
        self.a_p = a_p
        self.a_u = a_u
        self.gen_name = gen_name

    def get_adversary_utility(self):
        private_models = []
        utility_models = []
        # test under weak adversary and utility provider
        private_models.append(keras.models.load_model(os.path.join(self.main_dir, "adversaries_and_utility",self.dataset,'weak-adversary')))
        utility_models.append(keras.models.load_model(os.path.join(self.main_dir, "adversaries_and_utility",self.dataset,'weak-utility')))
        # test under strong adversary and utility provider
        private_models.append(keras.models.load_model(os.path.join(self.main_dir, "adversaries_and_utility",self.dataset,str.lower(self.gen_name)+'-strong-adversary')))
        utility_models.append(keras.models.load_model(os.path.join(self.main_dir, "adversaries_and_utility",self.dataset,str.lower(self.gen_name)+'-strong-utility')))
        # adversary and utility which was dynamically trained during the recent optimization
        private_models.append(self.a_p)
        utility_models.append(self.a_u)
        return private_models, utility_models

    def get_results(self, original_data, private_labels, utility_labels, lambda_p):
        private_data = self.generator.predict([original_data])
        if self.dataset == "MNIST" or self.dataset == "FashionMNIST":
            # Display a 2D manifold of the digits of Testing Datset
            Draw.original_vs_private(self.main_dir, original_data, private_data,lambda_p, self.dataset, self.gen_name)
        private, utility = self.get_adversary_utility()
        # Adversaries result
        private_acc, private_auroc = [], []
        for classifier in private:
            predicted = np.argmax(classifier.predict(private_data), axis = 1)
            actual = np.argmax(private_labels, axis = 1)
            private_acc.append(accuracy_score(actual, predicted))
            if self.dataset == "FashionMNIST":
                private_auroc.append(roc_auc_score(utils.to_categorical(actual), utils.to_categorical(predicted), multi_class = 'ovr', average = 'macro'))
            else:
                private_auroc.append(roc_auc_score(actual, predicted, average = 'macro'))
        # Utilities result
        utility_acc, utility_auroc = [], []
        for classifier in utility:
            predicted = np.argmax(classifier.predict(private_data), axis = 1)
            actual = np.argmax(utility_labels, axis = 1)
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
        with open(os.path.join(self.main_dir,"results",self.dataset,str(self.gen_name)+"-"+str(lambda_p)+"-private_acc.txt"), "a+") as file:
            file.write(str(max_private_acc)+"\n")
        with open(os.path.join(self.main_dir,"results",self.dataset,str(self.gen_name)+"-"+str(lambda_p)+"-utility_acc.txt"), "a+") as file:
            file.write(str(max_utility_acc)+"\n")
        with open(os.path.join(self.main_dir,"results",self.dataset,str(self.gen_name)+"-"+str(lambda_p)+"-private_auroc.txt"), "a+") as file:
            file.write(str(max_private_auroc)+"\n")
        with open(os.path.join(self.main_dir,"results",self.dataset,str(self.gen_name)+"-"+str(lambda_p)+"-utility_auroc.txt"), "a+") as file:
            file.write(str(max_utility_auroc)+"\n")
