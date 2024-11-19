import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random

def set_seed(random_seed):
    """Function for setting random seed for the relevant libraries."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    print(f"Random seed set as {random_seed}\n")


def plot_metrics(hist_dict, results_folder, date, type):
    """Function for plotting the training and validation results."""
    epochs = range(1, len(hist_dict['tr_loss'])+1)
    plt.plot(epochs, hist_dict['tr_loss'], 'g', label='Training loss')
    plt.plot(epochs, hist_dict['val_loss'], 'b', label='Validation loss')
    plt.title(f'{type}: training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_folder + '/' + date + '_' + type + '_tr_val_loss.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_acc'], 'g', label='Training accuracy')
    plt.plot(epochs, hist_dict['val_acc'], 'b', label='Validation accuracy')
    plt.title(f'{type}: training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(results_folder + '/' + date +  '_' + type + '_tr_val_acc.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(epochs, hist_dict['tr_f1'], 'g', label='Training F1 score')
    plt.plot(epochs, hist_dict['val_f1'], 'b', label='Validation F1 score')
    plt.title(f'{type}: training and validation F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig(results_folder + '/' + date +  '_' + type + '_tr_val_f1.jpg', bbox_inches='tight')
    plt.close()
