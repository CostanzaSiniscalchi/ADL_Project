# ADL_Project
Roshan Kenia, Costanza Sin

## Directory Structure & Key Files

### Data Collection

Under ```data_installation_scripts```, you will find scripts that download the PREVENT-AD Dataset from the web. We specified certain parameters in this script that allowed us to only acquire patients that were presymptomatic and had at least three scans in a 3-5 year period. 


### Data Preprocessing

Our preprocessing scripts are in the ```preprocessing folder``` which include skull stripping and converting the scans to a numpy format. 


### Subset of Data
After processing the data, we extracted the scans in numpy format and stored in the ```data``` folder. Since our dataset is very large, we included a subset of our data under ```stripped_3_scans_slices``` and ```stripped_5_scans_slices```. These folders contain 1 patient and either three or five longitudunal scans. 

This folder also contains metadata provided by PREVENT-AD about each of the patients. 

### Model

Under the ```Autoencoder``` directory, you will find code related to training and evaluating the random masked reconstruction model (SSL) and next scan prediction model. 

```model.py``` defines our ```ViTAutoEnc``` model. Data loader for splitting (train, val, test) and loading our slices of scans is provided in ```dsta_loader_ssl.py```. Our SSL task uses ```MRISliceDataLoader``` and our prediction task uses ```MRISliceGeneratorDataLoader```. 

### SSL

Training code for the ssl task is provided in ```train_ssl_slice.py```. Script for hyperparameter tuning is provided in ```train_ssl_slice_tuning.py```. 

### Prediction 

Training code for the prediction task is provided in ```train_prediction.py```. Script for hyperparameter tuning is provided in ```train_prediction_tuning.py```.

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
3. SSL Task: ```python train_ssl_slice.py```
4. Prediction Task: ```python train_prediction.py```
3. Evaluation: ```python test_ssl_pred.py```








