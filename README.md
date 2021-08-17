## Cyberbullying detection in text using machine learning and deep learning (v 1.0)
This repository contains python code to detect cyberbullying in text using machine learning algorithms like Support Vector Machine  (SVM), deep learning models like GRU + GloVe, and RoBERTa with multi-layer perceptron on it.

## Environment
This notebook is based on python 3.0+. Most of the library comes pre-installed with Google Colab. Rest required libraries can be installed by running the first block of the code.

## How to run the code
### TODO
1. Open the notebook in Google Colab. Here is a short <a href="https://www.youtube.com/watch?v=inN8seMm7UI">video</a> tutorial about how to use Google Colab or if you prefer reading a blog then please visit this <a href="https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c">link</a> to learn about Google Colab.
2. Once the notebook is open, you need to install the packages required for the project. Simply run the first block of the notebook, it will install all required libraries. The second block of the notebook will import all libraries required for the project.
3. After this you may need to download the data using this <a href="https://data.mendeley.com/datasets/jf4pzyvnpj/1">link</a>. Unpack this data and upload it to your Google Drive. After unpacking, size of data becomes 171 Mb that's why it cannot be uploaded to Github.
4. You may need to change the path of data in the notebook. Replace the path mentioned in the notebook with your data path. Path variables look like following in the notebook
<img src="https://github.com/girijesh97/LU_RM_Project/blob/master/images/rm_img_15.png" alt="Variable Path" align="center"  width="450" />
6. Also, this notebook requires GloVe 300d embedding, to download this embedding please visit this <a href="https://www.kaggle.com/thanakomsn/glove6b300dtxt">link</a>. Download the GloVe weight from the link and upload it to your Google Drive. You may need to change the path of variable `glove_path`. Assign the google drive path of your glove model in the variable `glove_path. After this simply run all the blocks of the notebook to reproduce the results.


# Project Description

## Workflow
#### Following diagram shows a complete workflow.
<img src="https://github.com/girijesh97/LU_RM_Project/blob/master/images/rm_img_12.png" alt="Complete Workflow" align="center"  width="300" />

## Introduction
This repo contains a `Lakehead_RM_Project` notebook which contains code for the following work.
1. Data
2. Data Preprocessing
3. Data Analysis and plotting
4. Model Building
5. Model Evaluation
6. Results

## Dataset
The dataset which has been used for this project can be found at this <a href="https://data.mendeley.com/datasets/jf4pzyvnpj/1">websit</a>. The size of the zipped dataset is `64 mb` which contains 8 different CSV in it. There are 5 columns out of which we will be using `Text` and `oh_label` for our analysis and modelling purpose.


## Data Pre-processing
After merging all 8 files, a single `data frame was created which has 448880 rows. But this data has many duplicate rows and blank rows along with other anomalies. The following list shows the data processing steps.
1. Converted all text to lower.
2. Fixed contraction like `isn't` to `is not from the text.
3. Removed hyperlink from the text.
4. Removed punctuations from the text.
5. Remove single characters except `a`.
6. Removed all Non-ASCII characters from text.
7. Trimmed extra space from the text.
8. Removed stopwords from the text.
9. Balanced output label counts.
## Data Analysis
1. Created Word Cloud to see the most frequently occurring words with and without stopwords.
2. Looked out for profanity in each sentence and plotted a bar graph to see how many sentences contain profanity in it.
3. Analyzes maximum and a minimum length of sentence to create an effective model.

## Model Building
#### 1. Support Vector Machine (SVM)
Created Linear SVM and Kernel SVM as baseline machine learning to check the performance of machine learning model on text data. Both models use the following approach:
* Vectorized data using TF-IDF mechanism
* Split data into train, test and validation set
* Trained both models using Scikit learn library.
Following images shows difference between Linear and Kernel SVM hypeplane

<img src="https://github.com/girijesh97/LU_RM_Project/blob/master/images/rm_img_06.png" alt="Linear and kernel SVM" align="center"  width="300" />                                                                                                                                      

#### 2. GRU with GloVe Embedding
GRU is a type of recurrent neural network (RNN) that works great with sequences like text.
* It can learn the long sequence of text with its special gates.
* It's well known for the understanding context of a sentence by remembering past information present in the sentence by using its gates.
* GloVe embedding stands for Global Vectors and it is a count-based, unsupervised learning model that captures both global statistics and local statistics of a corpus, to model the vector representation of words.
#### Following image shows a block of GRU unit.
<img src="https://github.com/girijesh97/LU_RM_Project/blob/master/images/rm_img_07.PNG" alt="GRU architecture" align="center"  width="300" />

#### 3. RoBERTa and Multi-Layer Perceptron (MLP)
A bidirectional Encoder Representation that uses transformers as its base architecture.
* It helps to learn and predict hidden patterns in the text.
* Modification of the key hyperparameters of BERT, which includes removing the next sentence prediction objective.
* To achieve even more appropriate classification results, MLP has been added on top of RoBERTa.
#### Following image represents architecture of Transformers
<img src="https://github.com/girijesh97/LU_RM_Project/blob/master/images/rm_img_10.PNG" alt="Transformers architecture" align="center"  width="300" />

## Model Evaluation
* _Data divison_ has taken place as follows:
1. 80% of Training data.
2. 10% Validation data during training.
3. 10%  Testing data.
* The standard size of the sentence is 150 words, however, Padding has been added to meet this average.
* GRU model has been trained for 10 epochs and RoBERTa has been trained for 5 epochs.
* We determine the effectiveness of the model, F-1 Score, accuracy, precision and recall.

## Result
Following accuracy, precision, recall and F-1 score were obtained on a different model

Model    | Linear SVM | Kernel SVM | GRU + GloVe | RoBERTa + MLP  
---      | ---        | ---        | ---         |---            
Accuracy | 0.856      |  0.855     | 0.841       | 0.899      
Precision| 0.859      |  0.856     | 0.849       | 0.875        
Recall   | 0.858      |  0.855     | 0.831       | 0.831          
F-1 Score| 0.858      |  0.855     | 0.836       | 0.88
