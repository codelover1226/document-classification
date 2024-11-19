from __future__ import print_function
from __future__ import division
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFile
from pathlib import Path

import utils

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}\n")

parser = argparse.ArgumentParser('Arguments for model training and validation')

parser.add_argument('--tr_data_folder', type=str, default="./data/train/",
                    help='path to training data with faulty images')
parser.add_argument('--val_data_folder', type=str, default="./data/validation/",
                    help='path to validation data with faulty images')
parser.add_argument('--results_folder', type=str, default="./results",
                    help='Folder for saving training results.')
parser.add_argument('--save_model_path', type=str, default="./models",
                    help='Path for saving model file.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size used for model training. ')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Base learning rate.')
parser.add_argument('--device', type=str, default='cpu',
                    help='Defines whether the model is trained using cpu or gpu.')
parser.add_argument('--num_classes', type=int, default=5,
                    help='Number of classes used in classification.')
parser.add_argument('--num_epochs', type=int, default=5,
                    help='number of training epochs for the classification head')
parser.add_argument('--unfreeze_epochs', type=int, default=5,
                    help='number of training epochs for the entire model')
parser.add_argument('--random_seed', type=int, default=8765,
                    help='Number used for initializing random number generation.')
parser.add_argument('--early_stop_threshold', type=int, default=2,
                    help='Threshold value of epochs after which training stops if validation accuracy does not improve.')
parser.add_argument('--date', type=str, default=time.strftime("%d%m%Y"),
                    help='Current date.')

args = parser.parse_args()

# PIL settings to avoid errors caused by truncated and large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# List for saving the names of damaged images
damaged_images = []
 
def get_datapaths():
    data_tr = []
    labels_tr = []
    data_val = []
    labels_val = []
    for i in range(1, args.num_classes + 1):
        tr_data = list(Path(args.tr_data_folder + str(i)).glob('*')) 
        val_data = list(Path(args.val_data_folder + str(i)).glob('*')) 
        tr_labels = [i-1] * len(tr_data)
        val_labels = [i-1] * len(val_data)
        data_tr += tr_data
        labels_tr += tr_labels
        data_val += val_data
        labels_val += val_labels
      
        print(f'Training samples in class {i}: {len(tr_data)}')
        print(f'Validation samples in class {i}: {len(val_data)}\n')

    data_dict = {'tr_data': data_tr, 'tr_labels': labels_tr, 
                'val_data': data_val, 'val_labels': labels_val}

    return data_dict


class ImageDataset(Dataset):
    """PyTorch Dataset class is used for generating training and validation datasets."""
    def __init__(self, img_paths, img_labels, transform):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.img_labels[idx]
        except:
            # Image is considered damaged if reading the image fails
            damaged_images.append(img_path)
            return None
        
        image = self.transform(image)
            
        return image, label


def initialize_model():
    """Function for initializing pretrained neural network model (DenseNet121)."""
    model_ft = models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, args.num_classes)

    return model_ft


def collate_fn(batch):
    """Helper function for creating data batches."""
    batch = list(filter(lambda x: x is not None, batch))
 
    return torch.utils.data.dataloader.default_collate(batch)


data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # uses imagenet_stats for mean and std
        # https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670/2
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_conf_matrix(y_true, y_pred, epoch, stage):
    """Create confusion matrix from classification results."""
    conf_matrix = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true', display_labels=np.arange(1, args.num_classes + 1))
    plt.savefig(args.results_folder + '/' + args.date + '_' + stage + '_epoch_' + str(epoch) + '.jpg', bbox_inches='tight')
    plt.close()


def initialize_dataloaders(data_dict):
    """Function for initializing datasets and dataloaders."""
    # Train and validation datasets 
    train_dataset = ImageDataset(img_paths=data_dict['tr_data'], img_labels=data_dict['tr_labels'],  transform=data_transforms)
    validation_dataset = ImageDataset(img_paths=data_dict['val_data'], img_labels=data_dict['val_labels'], transform=data_transforms)
    # Train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    return {'train': train_dataloader, 'val': validation_dataloader}


def get_criterion(data_dict):
    """Function for generating class weights and initializing the loss function."""
    y = np.asarray(data_dict['tr_labels'])
    # Class weights are used for compensating the unbalance 
    # in the number of training data from the different classes
    class_weights=class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights=torch.tensor(class_weights, dtype=torch.float).to(args.device)
    print('\nClass weights: ', class_weights.tolist())
    # Cross Entropy Loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    return criterion


def get_optimizer(model, lr, epochs, n_steps):
    """Function for initializing the optimizer."""
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=n_steps, epochs=epochs)

    return optimizer, scheduler


def train_model(model, dataloaders, criterion, optimizer, scheduler, stage):
    """Function for model training and validation."""
    since = time.time()
    # Lists for saving train and validation metrics for each epoch
    tr_loss_history = []
    tr_acc_history = []
    tr_f1_history = []
    val_loss_history = []
    val_acc_history = []
    val_f1_history = []
    
    # Best F1 value and best epoch are saved in variables
    best_f1 = 0
    best_epoch = 0
    early_stop = False

    # Train / validation loop
    for epoch in tqdm(range(args.num_epochs)):
        all_preds = []
        all_labels = []
        
        print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            running_f1 = 0.0

            # Iterate over data in batch
            for inputs, labels in dataloaders[phase]:
                if dataloaders[phase] is None:
                    continue
                else:
                    inputs = inputs.to(args.device)
                    labels = labels.long().to(args.device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Track history only in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        # Model predictions of the image labels for the batch
                        _, preds = torch.max(outputs, 1)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    
                    all_preds += preds.tolist()
                    all_labels += labels.tolist()
                    # Get weighted F1 score for the results
                    precision_recall_fscore = precision_recall_fscore_support(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0)
                    f1_score = precision_recall_fscore[2]

                    # update statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_acc += accuracy_score(labels.tolist(), preds.tolist())
                    running_f1 += f1_score

            # Calculate loss, accuracy and F1 score for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase])
            epoch_f1 = running_f1 / len(dataloaders[phase])

            print('\nEpoch {} - {} - Loss: {:.4f} Acc: {:.4f} F1: {:.4f}\n'.format(epoch+1, phase, epoch_loss, epoch_acc, epoch_f1))
            
            # Validation step
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                val_f1_history.append(epoch_f1)
                if epoch_f1 > best_f1:
                    print('\nF1 score {:.4f} improved from {:.4f}. Saving the model.\n'.format(epoch_f1, best_f1))
                    # Model with best F1 score is saved
                    pytorch_model_path = os.path.join(args.save_model_path, 'densenet_' + args.date + '.pth')
                    torch.save(model, pytorch_model_path)
                    print('Model saved to ', pytorch_model_path)
                    model = model.to(args.device)
                    best_f1 = epoch_f1
                    best_epoch = epoch
                elif epoch - best_epoch > args.early_stop_threshold:
                    # terminates the training loop if validation accuracy has not improved
                    print("Early stopped training at epoch ", str(epoch +1))
                    # Set early stopping condition
                    early_stop = True
                    break  
            elif phase == 'train':
                tr_acc_history.append(epoch_acc)
                tr_loss_history.append(epoch_loss)
                tr_f1_history.append(epoch_f1)

        # Creates confusion matrix after each epoch
        get_conf_matrix(all_labels, all_preds, epoch, stage)

        # Break outer loop if early stopping condition is activated
        if early_stop:
            break
        # Take scheduler step
        if scheduler:
            scheduler.step(val_acc_history[-1])

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation F1 score: {:.4f}'.format(best_f1))
    # Returns model with the weights from the best epoch (based on validation accuracy)
    hist_dict = {'tr_acc': tr_acc_history, 
                 'val_acc': val_acc_history, 
                 'val_loss': val_loss_history,
                 'val_f1': val_f1_history,
                 'tr_loss': tr_loss_history,
                 'tr_f1': tr_f1_history}

    return hist_dict

def main():
    # Set random seed(s)
    utils.set_seed(args.random_seed)
    # Load image paths and labels
    data_dict = get_datapaths()
    # Initialize the model 
    model = initialize_model()
    # Print the model architecture
    #print(model)
    # Send the model to GPU (if available)
    model = model.to(args.device)
    print("\nInitializing Datasets and Dataloaders...")
    dataloaders_dict = initialize_dataloaders(data_dict)
    criterion = get_criterion(data_dict)
    n_steps = len(dataloaders_dict['train'])

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    if args.num_epochs > 0:
        for name, param in model.named_parameters():
            if name not in ["classifier.weight", "classifier.bias"]:
                param.requires_grad = False
        optimizer, scheduler = get_optimizer(model, args.lr*10, args.num_epochs, n_steps)
        # Train and evaluate model
        hist_dict = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, stage='part_1')
        utils.plot_metrics(hist_dict, args.results_folder, args.date, 'part_1')

    if args.unfreeze_epochs > 0:
        pytorch_model_path = os.path.join(args.save_model_path, 'densenet_' + args.date + '.pth')
        model = torch.load(pytorch_model_path)
        model = model.to(args.device)

        for param in model.parameters():
            param.requires_grad = True
        
        optimizer, scheduler = get_optimizer(model, args.lr, args.unfreeze_epochs, n_steps)
        # Train and evaluate model
        hist_dict = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, stage='part_2')
        utils.plot_metrics(hist_dict, args.results_folder, args.date, 'part_2')
        
    print('Damaged images: ', damaged_images)

if __name__ == '__main__':
    main()
