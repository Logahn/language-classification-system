# Language Classification System

This Jupyter notebook contains a system for classifying languages based on text data. It includes data preprocessing, feature engineering, correlation reduction, and machine learning model training.

## Modules and Packages Used

The following modules and packages were used in this notebook:

- Matplotlib
- Pickle
- Scikit-learn
- Numpy
- Pandas
- Math
- Collections
- Itertools
- Seaborn

## Dataset

The dataset used in this notebook is stored in a CSV file named 'language.csv'. The dataset was preprocessed to remove any missing data and convert the 'text' and 'language' columns to strings.

## Creating Set of Features

A set of features was created based on the text data. It includes word count, character count, word density, punctuation count, vowel and consonant character count, exclamation and question mark count, unique words count, repeat words count, and more.

## Correlation Reduction

Principal Component Analysis (PCA) was applied to reduce the correlation between the features.

## Machine Learning Model

A Decision Tree Classifier was trained on the dataset and used to predict the language of text data. The trained model was saved using Pickle. The accuracy score of the model is displayed in a confusion matrix.

## Usage

To use this system, you can run the code in the Jupyter notebook and provide your own text data to predict its language.

## Link

Google colab file is located at https://colab.research.google.com/drive/1M_zRJwISxTOL4SU2Yo9V4ZcQHzYqpozU
