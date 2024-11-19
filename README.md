# Document classification model training

This repository contains code that can be used for training a model to classify input documents into distinct classes.
Best results can be achieved with input that contains clearly distinguishable document formats, which differ structurally from 
each other. 

In the National Archives of Finland, the code has been used for training a model to classify documents relating to Finnish
inheritance taxation. These consist of a four-page form and an appendix, which can vary in length and format. The model was 
trained to classify these documents into five classes, with one class for each page of the form and a separate class for the 
document belonging to the appendix. With this data, the model was able to reach a high level of classification accuracy:

Class|Training samples|Validation samples|Validation accuracy
-|-|-|-
Page 1|3799|422|99.95%
Page 2|3799|422|99.95%
Page 3|3801|422|99.88%
Page 4|3801|422|99.98%
Appendix|4500|500|99.56%

The code can be used for training a model to classify a varying number of document types in the input data. However, the following
'rules of thumb' should be kept in mind:

- The smaller the number of distinct classes, the better classification results can be generally expected.
- The more dissimilar the documents belonging to different classes are from each other and the more similar with other documents
  belonging to the same class, the easier the classification task is for the model.
- The more there are training examples from each document class, the easier it will be for the model to learn to classify
  the documents correctly.

The code has been built using the Pytorch library, 
and the model training is done by fine-tuning an existing [Densenet neural network model](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html). 

The code is split into two files: 

- `train.py` contains the main part of the code used for model training
- `utils.py` contains utility functions used for example for plotting the training and validation metrics

## Running the code in a virtual environment

These instructions use a conda virtual environment, and as a precondition you should have Miniconda or Anaconda installed on your operating system. 
More information on the installation is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

#### Create and activate conda environment using the following commands:

`conda create -n doc_classification_env python=3.7`

`conda activate doc_classification_env`

#### Install dependencies listed in the *requirements.txt* file:

`pip install -r requirements.txt`

#### Run the training code 

When using the default values for all of the model parameters, the training can be initiated from the command line by typing

`python train.py`

The different model parameters are explained in more detail below.

## Model parameters

### Parameters related to training and validation data

By default, the code expects the following folder structure, where training and validation data for each document class/type is
placed in a separate folder named with a numeric value (1,2,3...):

```
├──doc_classification 
      ├──models
      ├──results 
      ├──data
      |   ├──train
      |   |   ├──1
      |   |   ├──2
      |   |   ├──3 ...
      |   └──validation
      |   |   ├──1
      |   |   ├──2
      |   |   ├──3 ...
      ├──train.py
      ├──utils.py
      └──requirements.txt
```

Parameters:
- `tr_data_folder` defines the folder where the training data is located. It is expected to contain subfolders for each document class, named using numeric values (1, 2, 3...). Default folder path is `./data/train/`.
- `val_data_folder` defines the folder where the validation data is located. It is expected to contain subfolders for each document class, named using numeric values (1, 2, 3...). Default folder path is `./data/validation/`.

The parameter values can be set in command line when initiating training:

`python train.py --tr_data_folder ./data/train/ --val_data_folder ./data/validation/`

The accepted input image file types are .jpg, .png and .tiff. Pdf files should be transformed into one of these images formats before used as an input to the model.

### Parameters related to saving the model and the training and validation results

The training performance is measured using training and validation loss, accuracy and F1 score (more information on the F1 score can be found for example [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)). The average of these values is saved each epoch, and the resulting values are plotted and saved in the folder defined by the user. The confusion matrix containing the classification accuracy scores for each class is also plotted and saved after each epoch.

The trained model is saved by default after each epoch when the validation F1 score improves the previous top score. The model is saved by default to the `./models`
folder as `densenet_date.pth`. Date refers to the current date, so that a model trained on 4.7.2023 would be saved as `densenet_04072023.pth`.

Parameters:
- `results_folder` defines the folder where the plots of the training an validation metrics (loss, accuracy, F1-score) are saved. Default folder path is `./results`.
- `save_model_path` defines the folder where the model file is saved. Default folder path is `./models`.

The parameter values can be set in command line when initiating training:

`python train.py --results_folder ./results --save_model_path ./models`

### Parameters related to model training

A Number of parameters are used for defining the conditions for model training. 

The code allows the fine-tuning of the base model to be performed in two stages. In the first stage, the parameters of the base model are frozen, so that the training impacts only the parameters of the final classification layer. In the second stage, all parameters of the model are 'unfrozen' for training. In the code, the `num_epochs` parameter defines the number of epochs used in the first stage of training, and `unfreeze_epochs` defines the number of epochs used in the second stage of training.

Learning rate defines how much the model weights are tuned after each iteration based on the gradient of the loss function. In the code, the `lr` parameter defines the learning rate for the second stage of training, while the learning rate used for the classification layer's parameters in the first training stage is automatically set to be 10 times larger than `lr`.

Number of document types/classes used in the classification task is set using the `num_classes` parameter. This should correspond with the number of data folders used for the training and validation data.

Batch size (`batch_size`) sets the number of images that are processed before the model weights are updated. Early stopping (`early_stop_threshold`) is a method used for reducing overfitting by stopping training after a specific learning metric (loss, accuracy etc.) has not improved during a defined number of epochs.

Random seed (`random_seed`) parameter is used for setting the seed for initializing random number generation. This makes the training results reproducible when using the same seed, model and data. 

The `device` parameters defines whether cpu or gpu is used for model training. Currently the code does not support multi-gpu training.

Parameters:
- `lr` defines the learning rate used for the second stage of training. The learning rate for the first training stage (classification layer) is 10 times larger. Default value for the base learning rate is `0.0001`.
- `batch_size` defines the number of images in one batch. Default batch size is `16`.
- `num_epochs` sets the number of epochs used in the first stage of training. Default value is `5`.
- `unfreeze_epochs` sets the number of epochs used in the second stage of training. Default value is `5`.
- `num_classes`: sets the number of document types/classes used in the classification task. Default value is `5`.
- `early_stop_threshold` defines the number of epochs that training can go on without improvement in the chosen metric (validation F1 score by default). Default value is `2`.
-  `random_seed` sets the seed for initializing random number generation. Default value is `8765`.
-  `device` defines whether cpu or gpu is used for model training. Value can be for example `cpu`, `cuda:0` or `cuda:1`, depending on the specific gpu that is used.

The parameter values can be set in command line when initiating training:

`python train.py --lr 0.0001 --batch_size 16 --num_epochs 5 --unfreeze_epochs 5 --num_classes 5 --early_stop_threshold 2 --random_seed 8765 --device cpu`
