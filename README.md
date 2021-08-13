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
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
```

## Miscellaneous Information
- Datasets : links

## Steps
1. The data is preprocessed already.
2. Create 2 DF's, Train, Test Set with and without one hot encoding.
3. Fit the Train sets in to the model for a baseline accuracy
4. Tune the hyperparameters of the RandomForestClassifier & Gridsearch
5. Look for the best parameters.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
