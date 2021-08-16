## Cyberbullying detection in text using machine learning and deep learning (v 1.0)
This repository contains python code to detect cyberbullying in text using machine learning algorithms like Support Vector Machine  (SVM), deep learning models like GRU + GloVe, and RoBERTa.

## Introduction
This repo contains python notebook.
Notebook 1: This notebook contains code for following work.
1. Data Loading
2. Data Preprocessing
3. Data Analysis and plotting
4. Model Building
5. Model Evaluation

## Environment
This notebook is based on python 3.0+. Most of the library comes pre-installed with Google Colab. Rest required libraries can be installed by running first block of the code.

## Dataset
The dataset which has been used for this project can be found at (https://data.mendeley.com/datasets/jf4pzyvnpj/1). Size of dataset is `64 mb` which contains 8 different csv in it. There are ------ columns out of which we will be using `Text` and `oh_label` for our analysis and modelling purpose.

#### NOTE: Data Processing could take lot of time so I have saved a copy of processed data. It can be found at this link (----------------------------).

## Data Pre-processing
After merging all 8 files, a single `dataframe` was created which has ----- number of rows. But this data has many duplicates rows and balnk rows aliong with other anamolies. Following list shows the data processing steps.
1. Converted all text to lower.
2. Fixed contraction like `isn't` to `is not` from text.
3. Removed hyperlink from text.
4. Removed punctuations from text.
5. Remove single characters except `a`.
6. Removed all Non ASCII characters from text.
7. Trimmed extra space from text.
8. Removed stopwords from the text.
9. Balanced output label counts. 
## Data Analysis
1. Created Word Cloud to see most frequently occuring words with and without stopwords.
2. Looked out for profanity in each sentence and plotted a bar graph to see how many sentences contains profanity in it.
3. Analyzes maximum and minimum length of sentence to creat an effective model.
## Model Building
#### 1. Support Vector Machine (SVM)
Created Linear SVM and Kernel SVM as a baseline machine learning to check performance of machine learning model on text data. Both model uses following approach:
a. Vectorized data using TF-IDF mechanism
b. Split data into train, test and validation set
c. Train both model using Scikit learn library.


