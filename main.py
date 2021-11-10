#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time

from A1.A1 import A1
from A2.A2 import A2
from B1.B1 import B1
from B2.B2 import B2
from helper.helper import (extract_features_labels, prepare_data,
                           prepare_data_for_cnn, prepare_data_from_csv)

# # ======================================================================================================================
# # Constants and paths
datasetList = ['celeba', 'cartoon_set'] # a list of the exact folder names of the datasets
celeb = datasetList[0]
cartoon = datasetList[1]

taskList = ['A1', 'A2', 'B1', 'B2'] # a list of the exact folder names of the tasks
gender = taskList[0]
emotion = taskList[1]
face_shape = taskList[2]
eye_color = taskList[3]

home_dir = os.path.abspath(os.curdir) # the current directory
log_path = os.path.join(home_dir, 'helper', 'base_log.log') # the path of log file

# # ======================================================================================================================
# # The configuration of logging module (feel free to change the logging level)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_path,
                    filemode='w')

# # ======================================================================================================================
# # Main program
logging.info('*****************************************')
logging.info('**********[Main program starts]**********')
logging.info('*****************************************\n')

start_time = time.time()

# # ======================================================================================================================
# # Task A1
logging.info('**********[Task A1 starts]**********')

# Prepare training and testing set from the source of data
x_train, x_test, y_train, y_test = prepare_data(gender, celeb)

# Or prepare training and testing set from the existing csv file
# which is extacted from the data source for convenience of model training
# x_train, x_test, y_train, y_test = prepare_data_from_csv(
#     os.path.join(home_dir, gender, 'A1_dataset.csv'))

model_A1 = A1(x_train, x_test, y_train, y_test)
acc_A1_train, clf = model_A1.train()
acc_A1_test = model_A1.test(clf)

##### for debugging only #####
# print('TA1:{},{}'.format(acc_A1_train, acc_A1_test))

logging.info('**********[Task A1 ends]**********')
logging.info('***********************************')
# # ======================================================================================================================
# # Task A2
logging.info('**********[Task A2 starts]**********')

# Prepare training and testing set from the source of data
x_train, x_test, y_train, y_test = prepare_data(emotion, celeb)

# Or prepare training and testing set from the existing csv file
# which is extacted from the data source for convenience of model training
# x_train, x_test, y_train, y_test = prepare_data_from_csv(
#     os.path.join(home_dir, emotion, 'A2_dataset.csv'))

model_A2 = A2(x_train, x_test, y_train, y_test)
acc_A2_train, clf = model_A2.train()
acc_A2_test = model_A2.test(clf)

#### for debugging only #####
# print('TA2:{},{}'.format(acc_A2_train, acc_A2_test))

logging.info('**********[Task A2 ends]**********')
logging.info('***********************************')
# # ======================================================================================================================
# # Task B1
logging.info('**********[Task B1 starts]**********')

# Prepare training, validation, evaluation and testing set from the source of data
train_gen, valid_gen, eval_gen, test_gen = prepare_data_for_cnn(
    face_shape, cartoon)

model_B1 = B1(train_gen, valid_gen, eval_gen, test_gen)
acc_B1_train = model_B1.train()
acc_B1_test = model_B1.test(os.path.join(
    home_dir, face_shape, 'B1_CNN_model.h5'))

#### for debugging only #####
# print('TB1:{},{}'.format(acc_B1_train, acc_B1_test))

logging.info('**********[Task B1 ends]**********')
logging.info('***********************************')
# # ======================================================================================================================
# # Task B2
logging.info('**********[Task B2 starts]**********')

# Prepare training, validation, evaluation and testing set from the source of data
train_gen, valid_gen, eval_gen, test_gen = prepare_data_for_cnn(
    eye_color, cartoon)

model_B2 = B2(train_gen, valid_gen, eval_gen, test_gen)
acc_B2_train = model_B2.train()
acc_B2_test = model_B2.test(os.path.join(
    home_dir, eye_color, 'B2_CNN_model.h5'))

#### for debugging only #####
# print('TB2:{},{}'.format(acc_B2_train, acc_B2_test))

logging.info('**********[Task B2 ends]**********')
logging.info('***********************************')
# # ======================================================================================================================
# # Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

end_time = time.time()
elapsed_time = end_time - start_time

logging.info(
    f'[Total execution time]: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

logging.info('*****************************************')
logging.info('***********[Main program ends]***********')
logging.info('*****************************************\n')
