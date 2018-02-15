import numpy as np


def test_cat(train_set_y, classes):
    # Example of a picture
    index = 2
    assert (train_set_y[:, index] == [1])


def test_not_cat(train_set_y, classes):
    # Example of a picture
    index = 1
    assert (train_set_y[:, index] == [0])

