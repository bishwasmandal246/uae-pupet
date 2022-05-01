import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Get final results of PUPET')
parser.add_argument('-d', '--dataset', type=str, metavar='', required=True, help = 'Dataset name: MNIST, FashionMNIST, UCIAdult, USCensus or Data-Type-Aware (for UCI Adult)')
args = parser.parse_args()

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))
    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points
    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return upper[:-1] # For our case we don't need lower hull so we only return the upper convex hull

def upt_curve(UAE, AE, VAE, bVAE, main_dir, dataset):
    # Guess to get highest accuracy by outputting same class which has highest frequency
    if dataset == "MNIST" or dataset =="USCensus":
        guess_x = 0.5 #private accuracy
        guess_y = 0.5 #utility accuracy
    elif dataset == "UCIAdult":
        guess_x = 0.67
        guess_y = 0.75
    elif dataset == "FashionMNIST":
        guess_x = 0.1
        guess_y = 0.5
    else:
        print("Dataset not recognized!")
        sys.exit(0)
    UAE[0].append(guess_x)#private accuracy
    UAE[2].append(guess_y)#utility accuracy
    AE[0].append(guess_x)
    AE[2].append(guess_y)
    VAE[0].append(guess_x)
    VAE[2].append(guess_y)
    bVAE[0].append(guess_x)
    bVAE[2].append(guess_y)
    # Make tuples of X and Y points (For private and utility accuracies)
    uae_tuples, ae_tuples, vae_tuples, bvae_tuples = [], [], [], []
    for i in range(len(UAE[0])):
        uae_tuples.append((UAE[0][i], UAE[2][i]))
        ae_tuples.append((AE[0][i], AE[2][i]))
        vae_tuples.append((VAE[0][i], VAE[2][i]))
        bvae_tuples.append((bVAE[0][i], bVAE[2][i]))
    # Get the upper convex hull points
    uae = convex_hull(uae_tuples)
    ae = convex_hull(ae_tuples)
    vae = convex_hull(vae_tuples)
    bvae = convex_hull(bvae_tuples)
    # to draw joining lines to the best possible guess
    uae.append((guess_x, guess_y))
    ae.append((guess_x, guess_y))
    vae.append((guess_x, guess_y))
    bvae.append((guess_x, guess_y))
    # matplotlip to draw utility-privacy curve (UPT-Curve)
    fig, ax1 = plt.subplots(figsize=(8,6))
    # Guess point
    ax1.scatter(guess_x, guess_y, color='red', s = 80, label="Best Guess", marker="X", alpha = 0.5)
    # UAE
    ax1.scatter(UAE[0][:-1], UAE[2][:-1], color='C1', s = 80, label="UAE-PUPET", marker="o", alpha = 0.5)
    ax1.plot(*zip(*uae), color='C1', linestyle='-.', markersize=20)# draw lines on convex hull upper pairs only
    # AE
    ax1.scatter(AE[0][:-1], AE[2][:-1], color='C2', s = 80, label="AE-PUPET", marker="o", alpha = 0.5)
    ax1.plot(*zip(*ae), color='C2', linestyle='-.', markersize=20)
    # VAE
    ax1.scatter(VAE[0][:-1], VAE[2][:-1], color='C3', s = 80, label="VAE-PUPET", marker="o", alpha = 0.5)
    ax1.plot(*zip(*vae), color='C3', linestyle='-.', markersize=20)
    # b-VAE
    ax1.scatter(bVAE[0][:-1], bVAE[2][:-1], color='C4', s = 80, label="b-VAE-PUPET", marker="o", alpha = 0.5)
    ax1.plot(*zip(*bvae), color='C4', linestyle='-.', markersize=20)
    
    ax1.grid()
    ax1.legend(fontsize = 14)
    ax1.set_title('UPT-curve', fontsize = 14)
    ax1.set_xlabel("Private Feature acc scores", fontsize = 14)
    ax1.set_ylabel('Utility Feature acc scores', fontsize = 14)
    ax1.tick_params(labelsize=14, pad = 8)

    #Save the plot
    fig.savefig(os.path.join(main_dir,"results","UPTCurves",str(dataset)+'-UPT-curve.png'), dpi = 500, bbox_inches='tight')

if __name__ == "__main__":
    main_dir = "uae-pupet"
    # extract all infromation for the final results from "results" folder
    result_dir = os.path.join(main_dir, "results", args.dataset)
    # to store all results: Private and Utility accuracy and auroc
    UAE = [[], [], [], []]
    VAE = [[], [], [], []]
    bVAE = [[], [], [], []]
    AE = [[], [], [], []]
    # iterators over generator
    generators = ["UAE-", "VAE-", "AE-", "b-VAE-"]
    # iterators for different metrics
    metrics = ["-private_acc.txt", "-private_auroc.txt", "-utility_acc.txt", "-utility_auroc.txt"]
    for generator in generators:
        for j, metric in enumerate(metrics):
            #Our lambda_p used in experiments are [0,10,20,..100] for data type unaware conditions.
            if args.dataset!="Data-Type-Aware":
                draw_upt = True
                min_lambda_p = 0
                max_lambda_p = 101
            else:
                draw_upt = False
                min_lambda_p = 10
                max_lambda_p = 11
            for lambda_p in range(min_lambda_p, max_lambda_p, 10):
                # precise file matching without regular expressions to store values in proper sructure i.e. from lambda_p = 0 to 100
                scores = []
                file = os.path.join(result_dir, generator+str(lambda_p)+metric)
                with open(file, "r") as reader:
                    line = reader.readline()
                    # read until EOF
                    while line != '':
                        scores.append(float(line))
                        line = reader.readline()
                if generator == "UAE-":
                    UAE[j].append(np.mean(np.array(scores)))
                elif generator == "VAE-":
                    VAE[j].append(np.mean(np.array(scores)))
                elif generator == "b-VAE-":
                    bVAE[j].append(np.mean(np.array(scores)))
                else:
                    AE[j].append(np.mean(np.array(scores)))
    
    print("Accuracy scores:\n")
    print(" -----------------------------------------------------------------------------------------------------------------------------")
    print("|    Lambda_P    |   UAE private   |   UAE utility  |     AE private    |    AE utility   |   b-VAE private |   b-VAE utility |")
    print(" -----------------------------------------------------------------------------------------------------------------------------")
    for i, j in enumerate(range(min_lambda_p, max_lambda_p, 10)):
        print("|     {:3d}        |      {:.4f}     |      {:.4f}    |      {:.4f}       |      {:.4f}     |      {:.4f}     |      {:.4f}     |".format(j, UAE[0][i], UAE[2][i], AE[0][i], AE[2][i], bVAE[0][i], bVAE[2][i]))
    print(" -----------------------------------------------------------------------------------------------------------------------------")
    print("\nAUROC scores:\n")
    print(" -----------------------------------------------------------------------------------------------------------------------------")
    print("|    Lambda_P    |   UAE private   |   UAE utility  |     AE private    |    AE utility   |   b-VAE private |   b-VAE utility |")
    print(" -----------------------------------------------------------------------------------------------------------------------------")
    for i, j in enumerate(range(min_lambda_p, max_lambda_p, 10)):
        print("|     {:3d}        |      {:.4f}     |      {:.4f}    |      {:.4f}       |      {:.4f}     |      {:.4f}     |      {:.4f}     |".format(j, UAE[1][i], UAE[3][i], AE[1][i], AE[3][i], bVAE[1][i], bVAE[3][i]))
    print(" -----------------------------------------------------------------------------------------------------------------------------")
    if draw_upt:
        upt_curve(UAE,AE,VAE,bVAE, main_dir, args.dataset)
