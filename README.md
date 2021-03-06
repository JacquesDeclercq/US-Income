# US-Income Project

Predict the income of a US citizen based.

## Mission objectives

- Be able to analyze a machine learning problem
- Be able to reason about possible causes of overfitting
- Be able to remedy the causes of overfitting
- Be able to tune parameters of a machine learning model
- Be able to write clean and documented code.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries.

## Usage

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('seaborn-whitegrid')
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from yellowbrick.model_selection import validation_curve 
```

## Miscellaneous Information
- Datasets : - [../../../additional_resources/datasets/US Income/cleaned/data_train.csv](../../../additional_resources/datasets/US%20Income/cleaned/data_train.csv)..
- [../../../additional_resources/datasets/US Income/cleaned/data_test.csv](../../../additional_resources/datasets/US%20Income/cleaned/data_test.csv)

## Workflow
1. The data is preprocessed already.
2. Create 2 DF's, Train, Test Set with and without one hot encoding.
3. Fit the Train sets in to the model for a baseline accuracy
4. Tune the hyperparameters of the RandomForestClassifier, Gridsearch & RandomsearchCV
5. Look for the best parameters.

### Step 1
The data has been cleaned already. Last check for NaN Values


<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.15.11.png" width="450">

### Step 2
Create the Train & Test set with and without one-hot-encoding


<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.15.27.png" width="450">

### Step 3
1. Our Baseline Accuracy

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.05.11.png" width="450">

2. Confusion Matrix

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.05.30.png" width="450">

3. Cross Validation Score & Curve

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.05.40.png" width="450">

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/crossvalidationcurve.png" width="450">


6. Feature Importance

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.05.47.png" width="450">



### Step 4
These are the parameters we hyper-tuned :

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.06.14.png" width="450">

And these are the results of our tuned model with Gridsearch & n_job : -1 (we used Kfold : 3)

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.06.41.png" width="450">
<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.06.58.png" width="450">

### Step 4

And these are the results of our tuned model with Gridsearch & without n_job : -1 (we used Kfold : 10)

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.07.28.png" width="450">
<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.07.44.png" width="450">

As u can see these parameters gave us the best score for the train and test set.
<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.07.10.png" width="450">


### Step 4

With RandomsearchCV (Computation time is significantly reduced)

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.07.54.png" width="450">

But gave us different parameters as best combination :

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.08.02.png" width="450">

And a score of: 

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.08.10.png" width="450">

What about the accuracy, you say ?

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.08.31.png" width="450">

<img src="https://github.com/JacquesDeclercq/US-Income/blob/main/Images/Screenshot%202021-08-16%20at%2015.08.42.png" width="450">




## Conclusion
I'v kept the Train & Test set seperated, to make sure I don't cause a overfitting problem from the get-go. The OHE dataset/test didn't bring any better score. To avoid over-fitting in random forest, the main thing you need to do is optimize a tuning parameter that governs the number of features that are randomly chosen to grow each tree from the bootstrapped data. Typically, you do this via ????-fold cross-validation, where ????={5,10}, and choose the tuning parameter that minimizes test sample prediction error. In addition, growing a larger forest will improve predictive accuracy, although there are usually diminishing returns once you get up to several hundreds of trees. 


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
