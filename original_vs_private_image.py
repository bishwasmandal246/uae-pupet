import os
from numpy import random
import matplotlib.pyplot as plt

class Draw:
    def original_vs_private(main_dir, original, private, lambda_p, dataset, generator_type):
        '''
        plots the privatized images along with its original version and save it
        '''
        n = 15  # figure with 15x15 digits
        digit_size = 28
        plt.figure(figsize=(16,7))
        for i in range(48):
            img = random.randint(9999)
            orig = original[img].reshape(digit_size, digit_size)
            priv = private[img].reshape(digit_size, digit_size)
            plt.subplot(6,16,2*i+1)
            plt.axis('off')
            plt.imshow(orig, cmap = 'gray')
            plt.subplot(6,16,2*i+2)
            plt.imshow(priv, cmap = 'gray')
            plt.axis('off')
        plt.savefig(os.path.join(main_dir,"generated_images",str(generator_type)+"-"+str.lower(dataset)+"-"+str(lambda_p)), dpi=500, bbox_inches = 'tight')
