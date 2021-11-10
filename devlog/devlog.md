# Experiment Results

## Face Detection


|   | celeba | cartoon_set |
| - | - | - |
| dlib | 4833/5000 96.7% | 7941/10000 79.4% |
| haar | 4617/5000 92.3% | 2097/10000 21.0% |

## A1

- LinearSVM:
  - acc_train: 0.9353 / acc_val: 0.9205 / **acc_test: 0.9214** ('linearsvc__C': 0.1) / 00:00:24
  - acc_train: 0.9248 / acc_val: 0.9140 / acc_test: 0.9145 ('linearsvc__C': 0.1, with features selected) / 00:00:14
- RandomForest:
  - acc_train: 1.0 / acc_val: 0.8673 / acc_test: 0.8586 ('max_features': 7, 'n_estimators': 101) / 00:00:45
  - acc_train: 1.0 / acc_val: 0.8679 / acc_test: 0.8676 ('max_features': 10, 'n_estimators': 91, with features selected) / 00:00:38

## A2

- LinearSVM:
  - acc_train: 0.9072 / acc_val: 0.8871 / acc_test: 0.8952 ('linearsvc__C': 0.1) / 00:00:25
  - acc_train: 0.8981 / acc_val: 0.8889 / **acc_test: 0.8959** ('linearsvc__C': 1, with features selected) / 00:00:16
- RandomForest:
  - acc_train: 1.0 / acc_val: 0.8900 / acc_test: 0.8945 ('max_features': 9, 'n_estimators': 101) / 00:00:39
  - acc_train: 1.0 / acc_val: 0.8883 / acc_test: 0.8940 ('max_features': 10, 'n_estimators': 71, with features selected) / 00:00:34

## B1

- CNN:
  - acc_train: 0.9846 / acc_val: 0.9440 / acc_test: **0.9450** / 00:19:08

## B2

- CNN:
  - acc_train: 0.7923 / acc_val: 0.8110 / acc_test: **0.8027** / 00:15:40

# Task Description

- Task A1: Gender detection
- Task A2: Emotion detection
- Task B1: Face shape recognition
- Task B2: Eye color recognition

# Dataset Description

- A1/A2: celeba (5000 images)
- B1/B2: cartoon_set (10000 images)

# Report Structure

## Abstract 5` (a brief overview of the methodology/results, code link)

- Page 1
- background
- task overview
- methods used
- results
- code link

## Introduction 7` (the problem, a brief bird’s-eye view of the methodologies you adopted and the organization of this report)

- Page 1
- task introduction
- methods used in the flow
- report organization

## Literature Survey 15` (an overview of potential approaches to solve the tasks)

- Page 1-3
- face detection
  - dlib
  - haar
- modeling algorithm
  - svm
  - cnn

## Description of Models 8` (the model you are using for each task, along with the reason)

- Page 3-4

## Implementation 12` (the detailed implementation of your models)

- Page 4-6

### Datasets

- dataset description
- amount of images
- pixel of image
- fair

### Task A1: Gender detection

- compare the results of haar and dlib
- haar cascade shows a low detection ratio

### Task A2: Emotion detection

- hyposis of lip as the features
- feature selection proves it

### Task B1: Face shape recognition

- dlib shows a false-positive problem
- plot the 68 landmarks

### Task B2: Eye color recognition

## Experimental Results and Analysis 8` (the results)

- Page 6-7
- randomforest overfitting

## Conclusion 5` (the findings and directions for future improvements)

- Page 7-8

## References

- Page 8

# Misc

1. Kaggle,  **数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已**
