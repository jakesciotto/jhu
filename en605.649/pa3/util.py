# -----------------------------------------------------------
# util.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

from collections import Counter
from itertools import islice
import random
import csv 
 
def take(n, iterable):
    """
    Returns first n items from a dictionary as a list.
    """
    return list(islice(iterable, n))

def choose_at_random(dataset, n_instances):
    """
    Due to the size of the datasets, choosing n_instances at random.
    """
    return dataset.sample(n = n_instances)

def put_folds_together(fold0, fold1, fold2, fold3, fold4):
    """
    This method puts the folds together and returns training
    and testing sets. This is necessary to ensure that there
    is a hold out set.
    """
    training, testing = [], []

    testing.append(fold0)
    training.append(fold1 + fold2 + fold3 + fold4)

    testing.append(fold1)
    training.append(fold0 + fold2 + fold3 + fold4)

    testing.append(fold2)
    training.append(fold0 + fold1 + fold3 + fold4)

    testing.append(fold3)
    training.append(fold0 + fold1 + fold2 + fold4)

    testing.append(fold4)
    training.append(fold0 + fold1 + fold2 + fold3)

    return training, testing

def train_test_split(dataset, ratio): 
    """
    Split the data as appropriate by ratio.
    """
    train_num = int(len(dataset) * ratio) 
    train = [] 
    test = list(dataset) 
    # make sure the index is shuffled
    while len(train) < train_num: 
        index = random.randrange(len(test)) 
        train.append(test.pop(index)) 
    return train, test 

def cross_val(dataset, folds):
    """ 
    Perform cross-validation split of dataset before KNN.
    """
    split_data = list()
    copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        # shuffles the dataset
        while len(fold) < fold_size:
            index = random.randrange(len(copy))
            fold.append(copy.pop(index))
        split_data.append(fold)
    print("Fold size: ", fold_size)
    return split_data

def parse_file(filename):
    """
    Parse file into a dictionary for ID3, specifically.
    """
    data = []
    with open(filename, 'r') as file:
        csv_file = csv.reader(file)
        headers = next(csv_file)
        for row in csv_file:
            data.append(dict(zip(headers, row)))
    return data
 
def get_five_folds(instances):    
    """
    Gets five folds in a manual way.
    """
    fold0, fold1, fold2, fold3, fold4 = [[] for i in range(5)]
 
    random.shuffle(instances)
 
    classes = []
 
    for instance in instances:
        classes.append(instance['Class'])
 
    unique_classes = list(Counter(classes).keys())
 
    for unique in unique_classes:
        counter = 0
        for instance in instances:
            if unique == instance['Class']:
 
                if counter == 0:
                    fold0.append(instance)
                    counter += 1
                elif counter == 1:
                    fold1.append(instance)
                    counter += 1
                elif counter == 2:
                    fold2.append(instance)
                    counter += 1
                elif counter == 3:
                    fold3.append(instance)
                    counter += 1
                else:
                    fold4.append(instance)
                    counter = 0
 
    # shuffle said folds
    random.shuffle(fold0)
    random.shuffle(fold1)
    random.shuffle(fold2)
    random.shuffle(fold3)
    random.shuffle(fold4)
 
    return  fold0, fold1, fold2, fold3, fold4