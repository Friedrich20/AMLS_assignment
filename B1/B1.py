#!/usr/local/bin/python
# -*- coding:utf-8 -*-

from helper.helper import test_model_cnn, train_model_cnn


class B1:
    def __init__(self, train_gen, valid_gen, eval_gen, test_gen):
        """Initialise the class with the training, validation, evaluation and testing generators
        """
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.eval_gen = eval_gen
        self.test_gen = test_gen

    def train(self):
        """Train the CNN model
        """
        train_acc, model_path = train_model_cnn(
            'B1', self.train_gen, self.valid_gen, self.eval_gen)
        return train_acc

    def test(self, model_path):
        """Test the CNN model
        """
        test_acc = test_model_cnn(model_path, self.test_gen)
        return test_acc
