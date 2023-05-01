# -----------------------------------------------------------
# id3.py
# 
# Jake Sciotto
# EN605.649 Introduction to Machine Learning
# Johns Hopkins University
# Fall 2020
# -----------------------------------------------------------

from math import log 
from collections import Counter
from pprint import pprint

TREE = None
 
def id3(instances, default, *args):
    """
    The ID3 algorithm works by calculating entropy of the dataset
    and making classification decisions.
    """

    # piece of code for tests
    flag = None

    for arg in args:
        flag = arg

    if len(instances) == 0:
        return Node(default)
 
    classes = []

    # get the class for each class instance
    for instance in instances:
        classes.append(instance['Class'])
 
    # create a leaf if all the instances are from the same class
    if len(Counter(classes)) == 1 or len(classes) == 1:
        tree = Node(find_mode(instances))
        return tree

    # find the best attribute, otherwise
    else:
        best_attribute = get_most_informative(instances, flag)
        if flag:
            print("Best attribute: ", best_attribute)
        tree = Node(find_mode(instances))

        # most informative attribute 
        tree.attribute = best_attribute
 
        best_values = []

        # get the values of the best attribute
        for instance in instances:
            try:
                best_values.append(instance[best_attribute])
            except:
                no_best_attribute = True
    
        # turn this into a list
        tree.attribute_values = list(set(best_values))
     
        # split instances and separate subset for each best attribute value
        for attribute_value in tree.attribute_values:

            instances_subset = []
            for instance in instances:
                if instance[best_attribute] == attribute_value:
                    instances_subset.append(instance) 
            
            # recursively create a subtree
            subtree = id3(instances_subset, find_mode(instances))
            
            # initialize values of the subtree and keep track of parents
            subtree.instances_labeled = instances_subset
            subtree.parent_attribute = best_attribute
            subtree.parent_attribute_value = attribute_value 
 
            tree.children[attribute_value] = subtree

        return tree
 
 
def find_mode(instances, *args):
    """
    Find the most common class.
    """

    # piece of code for tests
    flag = None

    for arg in args:
        flag = arg

    classes = []

    for instance in instances:
        classes.append(instance['Class'])

    # TESTING
    if flag:
        print("Mode: ", Counter(classes).most_common(1)[0][0])

    return Counter(classes).most_common(1)[0][0]
 
def entropy(instances, attribute, attribute_value, flag):  
    """
    Performs the calculation of entropy in ID3 algorithm.
    """

    classes = []
 
    for instance in instances:
        if flag:
            if instance[attribute] == attribute_value:
                classes.append(instance['Class'])
        else:
            classes.append(instance['Class'])
    count = Counter(classes)
 
    # if they are all the same class, entropy is 0
    if len(count) == 1:
        return 0
    else:
        # keep track of the weighted sum of the probabilities
        entropy = 0
        for c, count_c in count.items():
            p = count_c / len(classes)
            entropy += p * (log(p, 2))
        return -entropy

def gain_ratio(instances, attribute, *args):
    """
    Calculating the gain ratio to determine which class to assign
    to an instance.
    """

    # piece of code for tests
    flag = None

    for arg in args:
        flag = arg

    prior_entropy = entropy(instances, None, None, False)
 
    values = []

    for instance in instances:
        values.append(instance[attribute])
    counter = Counter(values)
     
    # entropy if we decide to split
    entropy_on_split = 0

    # information gain 
    info_split = 0

    for attribute_value, attribute_value_count in counter.items():
        p = attribute_value_count / len(values)
        entropy_on_split += (p * entropy(instances, attribute, attribute_value, True))
        info_split += p * (log(p, 2))
     
    info_gain = prior_entropy - entropy_on_split

    # TESTING 
    if flag:
        print("Information gain: ", info_gain)

    info_split = -info_split
 
    gain_ratio = None
 
    if info_split != 0:
        gain_ratio = info_gain / info_split
    else:
        gain_ratio = -1000

    # TESTING
    if flag:
        print("Gain ratio: ", gain_ratio)

    return gain_ratio
 
def get_most_informative(instances, *args):
    """
    Returns the most informative attribute.
    """

    # piece of code for tests
    flag = None

    for arg in args:
        flag = arg

    # determine the max gain ratio
    selected = None
    max_gain_ratio = -1000

    # list of attribute names without class column
    attributes = [key for key, value in instances[0].items()]
    attributes.remove('Class')

    for attribute in attributes:
        # calculate the gain ratio and store that value
        gain = gain_ratio(instances, attribute, flag)
        # test for new max gain ratio
        if gain > max_gain_ratio:
            max_gain_ratio = gain
            selected = attribute
 
    return selected
 
def get_accuracy(tree, instances, *args):
    """
    Gets accuracy from validation set.
    """

    flag = None

    for arg in args:
        flag = arg

    num_correct = 0
 
    for instance in instances:
        if predict(tree, instance, flag) == instance['Class']:
            num_correct += 1
 
    return num_correct / len(instances)
 
def predict(node, test_instance, *args):
    """
    Make a prediction on a node based on a test instance.
    """

    flag = None

    for arg in args:
        flag = arg

    if len(node.children) == 0:
        return node.label
    else:
        attr_value = test_instance[node.attribute]
 
        # we have to go all the way down the branch
        if attr_value in node.children and node.children[attr_value].pruned == False:
            # TESTING 
            if flag:
                print("Prediction at leaf: ", predict(node.children[attr_value], test_instance))
            return predict(node.children[attr_value], test_instance, False)
        # otherwise, return most common class
        else:
            instances = []
            for attr_value in node.attribute_values:
                instances += node.children[attr_value].instances_labeled
            return find_mode(instances)

def prune(node, instances, *args):
    """
    The node that we are going to prune.
    """

    # piece of code for tests
    flag = None

    for arg in args:
        flag = arg

    global TREE
    TREE = node
 
    def prune_node(node, instances):
        # if this is a leaf node
        if len(node.children) == 0:
            prior_accuracy = get_accuracy(TREE, instances)
            node.pruned = True

            if node.pruned and flag:
                print("Accuracy of node pruned: ", prior_accuracy)
 
            # If no improvement in accuracy, no pruning
            if prior_accuracy >= get_accuracy(TREE, instances):
                node.pruned = False
            return
 
        for value, child_node in node.children.items():
            prune_node(child_node, instances)
 
        # finally prune when we reach base of recursion
        prior_accuracy = get_accuracy(TREE, instances)
        node.pruned = True
 
        if prior_accuracy >= get_accuracy(TREE, instances):
            node.pruned = False
 
    prune_node(TREE, instances)

def print_tree(node, depth = 0):
    """
    Prints the children of a tree.
    """
    for child, child_value in node.children.items():
        # early stopping for printing
        if depth == 3: 
            break
        else:
            if isinstance(child_value, Node):
                print(" " * depth + ("Child: " + child))
                print_tree(child_value, depth + 1)
            else:
                print(" " * depth + (child + " : " + child_value))

class Node:
    """
    Class Node for use in ID3 algorithm.
    """
    def __init__(self, label):
        self.attribute = None
        self.attribute_values = []
        self.label = label
        self.children = dict()

        self.parent = None
        self.parent_attribute_value = None

        self.pruned = False
        self.instances_labeled = []
