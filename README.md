# COMS 6998: ForeseeAD 
Roshan Kenia, Costanza Siniscalchi, Sanmati Choudhary

## Directory Structure & Key Files

### Data Collection

Under ```data_installation```, you will find scripts that download the PREVENT-AD Dataset from the web. We specified certain parameters in this script that allowed us to only acquire patients that were presymptomatic and had at least three scans in a 3-5 year period. 


### Data Preprocessing

Our preprocessing scripts are in the ```preprocessing``` directory which include skull stripping and converting the scans to a numpy format. 


### Data Examples
After processing the data, we extracted the scans in numpy format and stored them in the ```data``` folder. Since our dataset is very large, we included a subset of our data under ```stripped_3_scans_slices``` and ```stripped_5_scans_slices```. Each folder contains 1 patient and either three or five of their scans. Each of the scan folders contains preprocessed numpy arrays of the MRI slices. 

This folder also contains metadata provided by PREVENT-AD about each of the patients. 

### Model

Under the ```Autoencoder``` directory, you will find code related to training and evaluating the random masked reconstruction model (SSL) and next scan prediction model. 

```model.py``` defines our ```ViTAutoEnc``` model. Data loader for splitting (train, val, test) and loading our slices of scans is provided in ```data_loader_ssl.py```. 

Our SSL task uses ```MRISliceDataLoader``` and our prediction task uses ```MRISliceGeneratorDataLoader```. 

### SSL

Training code for the ssl task is provided in ```train_ssl_slice.py```. The ViTAutoEnc model is trained to reconstruct a partially or entirely masked scan using the
surrounding context. 

Script for hyperparameter tuning is provided in ```train_ssl_slice_tuning.py```. We use optuna to tune parameters such as Embedding Dimension, Number of Transformer Blocks (Depth), Number of Attention Heads, MLP Dimension, Learning Rate, Batch Size, and Weight Decay. 


### Prediction 

Training code for the prediction task is provided in ```train_prediction.py```. The pretrained ViTAutoEnc model from the SSL task is used for fine-tuning on this next scan prediction task. The model architecture is slightly modified with the addition of a decoder for generation. 

Script for hyperparameter tuning is provided in ```train_prediction_tuning.py```. We use optuna to tune parameters such as the depth of the decoder, learning rate, and batch size. 

### Evaluation 

The ```eval``` folder contains ```test_ssl_pred.py``` that loads in the best weights from both models (SSL & Prediction) and runs evaluation on the test set. Our evaluation metrics are defined in ```eval_utils.py``` including SSIM, MS-SIM and coverage. 

*Note: our model weights are not provided in this folder as their size exceeds 3GB. 

### Sample Visualizations

We also provided a few of our visualizations on the test set for both the ssl and prediction task under ```test_viz```. 

## Setup & Installation

1. ```pip install -r requirements.txt```
2. Follow these instructions for fsl installer: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FslInstallation(2f)MacOsX.html. 
We already provide ```fslinstaller.py```

## Running Training & Evaluation: 
1. SSL Task: ```python train_ssl_slice.py```
2. Prediction Task: ```python train_prediction.py```
3. Evaluation: ```python test_ssl_pred.py```








