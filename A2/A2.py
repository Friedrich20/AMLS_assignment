#!/usr/local/bin/python
# -*- coding:utf-8 -*-

from helper.helper import (plot_binary_confusion_matrix, plot_ROC,
                           select_features, train_model)
from sklearn.metrics import accuracy_score


class A2:
    def __init__(self, x_train, x_test, y_train, y_test):
        """Initialise the class with the training and testing set
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        """Train the model on the training set
        """
        # Train LinearSVM classifier without feature selection
        # acc_train, clf = train_model(self.x_train, self.y_train, 'A2', 'LinearSVM')

        # Train RandomForest classifier without feature selection
        # acc_train, clf = train_model(self.x_train, self.y_train, 'A2', 'RandomForest')

        # Perform feature selection
        self.x_train, self.x_test, self.y_train, self.y_test = select_features(
            self.x_train, self.x_test, self.y_train, self.y_test)

        # # Train LinearSVM classifier after feature selection
        acc_train, clf = train_model(
            self.x_train, self.y_train, 'A2', 'LinearSVM')

        # Train RandomForest classifier after feature selection
        # acc_train, clf = train_model(
        #     self.x_train, self.y_train, 'A2', 'RandomForest')

        return acc_train, clf

    def test(self, clf):
        """Test the model on the testing set
        """
        y_pred = clf.predict(self.x_test)
        acc = accuracy_score(self.y_test, y_pred)

        ##### for ploting only #####
        # plot_binary_confusion_matrix(clf, self.x_test, self.y_test, [
        #                              'Smiling', 'Not smiling'])
        # plot_ROC(clf, self.x_test, self.y_test)

        return acc
