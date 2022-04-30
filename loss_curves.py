import os
from numpy import random
import matplotlib.pyplot as plt

class Loss:
    def curves(self, main_dir,lambda_p, dataset, generator_type, generator_loss, private_loss, utility_loss, generator_loss_test, private_loss_test, utility_loss_test):
        '''
        plots the loss curves
        '''
        total = [i for i in range(1,len(generator_loss)+1)]
        fig, ax1 = plt.subplots(figsize=(9,6))
        ax1.plot(total, generator_loss, label = "Train Generator Loss")
        ax1.plot(total, generator_loss_test, label = "Validate Generator Loss")
        ax1.legend()
        ax1.grid()
        ax1.legend(fontsize = 10)
        ax1.set_title('Loss-curve', fontsize = 10)
        ax1.set_xlabel("Steps", fontsize = 10)
        ax1.set_ylabel('Loss', fontsize = 10)
        ax1.tick_params(labelsize=10, pad = 6)
        #Save the plot
        fig.savefig(os.path.join(main_dir,"results","LossCurves",str(dataset),str(generator_type)+"-"+str(lambda_p)+'-generator.png'), dpi = 500, bbox_inches='tight')
        
        fig, ax1 = plt.subplots(figsize=(9,6))
        ax1.plot(total, private_loss, label = "Train Private Loss")
        ax1.plot(total, utility_loss, label = "Train Utility Loss")
        ax1.plot(total, private_loss_test, label = "Validate Private Loss")
        ax1.plot(total, utility_loss_test, label = "Validate Utility Loss")
        ax1.legend()
        ax1.grid()
        ax1.legend(fontsize = 10)
        ax1.set_title('Loss-curve', fontsize = 10)
        ax1.set_xlabel("Steps", fontsize = 10)
        ax1.set_ylabel('Loss', fontsize = 10)
        ax1.tick_params(labelsize=10, pad = 6)
        #Save the plot
        fig.savefig(os.path.join(main_dir,"results","LossCurves",str(dataset),str(generator_type)+"-"+str(lambda_p)+'-classifiers.png'), dpi = 500, bbox_inches='tight')
        return 0
