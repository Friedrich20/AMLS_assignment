# README

This repo serves as the code implementation of the project assigned in the module Applied Machine Learning Systems ELEC0134.

# Description

In this project, we are required to tackle two pairs of tasks within the field of computer vision. The first pair of tasks deals with binary classification problem, covering gender detection (Task A1) and emotion detection (Task A2). The next one works on multiclass classification problem, cover- ing face shape recognition (Task B1) and eye color recogni- tion (Task B2) respectively. In this paper, traditional ma- chine learning techniques like support vector machine (SVM) with linear kernel and random forest are involved to solve these image classification problems, along with two face detection approaches known as dlib’s facial landmark detector and Haar feature-based cascade classifier. The us- age of these methods combined with a complete feature en- gineering process gives us good classification performance, where the accuracy score on testing datasets for Task A1 reaches 92.1% and Task A2 as 89.6%. In addition, Convolu- tional Neural Network (CNN), one of the state-of-the-art deep learning algorithms, has also shown its strong capabil- ity on these tasks which achieves the accuracy score of 94.5% to Task B1 and 80.3% to Task B2.

# Requirements

- Python 3.7+ (Python 3.7.9 is recommended which is the version used in the development.)
- macOS or Windows (For Windows users, please change '/' into '\\\\' in helper.py line 161.)
- Required modules
  - check *requirement.txt* to ensure all modules included have been installed
  - or run ```pip3 install -r requirements.txt``` in terminal to install with ease

# Usage

1. Switch to the root directory of the repo
2. Run ```python3 main.py``` in terminal to start the main program
3. Monitor the running process in *base_log.log* if you like (logging level is set to ```INFO``` by default, feel free to alter)

# Structure

```.
├── A1                      # the folder of Task A1, where model and temporary dataset for Task A1 could be found
│   ├── A1.py               # scripts for Task A1
├── A2                      # the folder of Task A2, where model and temporary dataset for Task A2 could be found
│   ├── A2.py               # scripts for Task A2
├── B1                      # the folder of Task B1, where model for Task B1 could be found
│   ├── B1.py               # scripts for Task B1
├── B2                      # the folder of Task B2, where model for Task B2 could be found
│   ├── B2.py               # scripts for Task B2
├── Datasets                # the folder of datasets, with content removed on purpose
│   ├── Remark.md
│   ├── cartoon_set
│   ├── cartoon_set_test
│   ├── celeba
│   └── celeba_test
├── README.md               # this file
├── devlog                  # the folder of some experiment related files, kept for debugging
│   ├── candidate_models    # (for debugging only) trained models
│   ├── csv_datasets        # (for debugging only) temporary datasets
│   ├── devlog.md           # (for debugging only) records of the experiment results
│   ├── logs                # (for debugging only) logs
│   └── pics                # (for debugging only) pictures used for the report
├── helper                  # the folder of some necessary files
│   ├── base_log.log        # the main log, generated once the main program srarts
│   ├── face_models         # the models used for face detection
│   └── helper.py           # core functions for face detection, feature engineering and model training
├── main.py                 # the entrance to main program
└── requirements.txt        # the list of required modules
```

# Having problems?

If you run into problems, please either file a github issue or send an email to uceewta@ucl.ac.uk.
