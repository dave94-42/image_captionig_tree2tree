from myCode.read_images_file import read_images
from myCode.read_sentence import label_tree_with_sentenceTree
import tensorflow as tf
from random import randrange, uniform
from random import shuffle
import numpy as np

######################
#functions to extract all the image trees and all the sentence trees

def get_image_batch(data):
    to_return = []
    for el in data:
        to_return.append(el["img_tree"])
    return to_return

def get_sentence_batch(data):
    to_return=[]
    for el in data:
        to_return.append(el['sentence_tree'])
    return to_return

def load_data(args):
    print('loading image trees....')
    train_data = read_images(args[0])
    val_data = read_images(args[1])
    print('loading sentence trees...')
    label_tree_with_sentenceTree(train_data,val_data, args[2])
    return train_data,val_data

def load_all_data(args,):
    print('loading image trees....')
    train_data = read_images(args[0])
    val_data = read_images(args[1])
    test_data = read_images(args[3])
    print('loading sentence trees...')
    label_tree_with_sentenceTree(train_data+val_data, test_data, args[2])
    return train_data+val_data,test_data

def laod_test_data(args,dictionary, embeddings):
    print('loading image trees....')
    test_data = read_images(args[0])
    print('loading sentence trees...')
    label_tree_with_sentenceTree(test_data,args[3],embeddings,dictionary)
    return test_data

#######################

def help():
    """
    function exlaning argumen to be passed
    :return:
    """
    print("1 -> train set file\n2 -> validation file\n3 -> embedding dictionaty\n4 -> tag dictionary\n"
          "5 -> parsed sentence dir")


def define_flags():
    """
    function that define flags used later in the traning
    :return:
    """
    tf.flags.DEFINE_string(
        "activation",
        default='tanh',
        help="activation used where there are no particular constraints")

    tf.flags.DEFINE_integer(
        "max_iter",
        default=105,
        help="Maximum number of iteration to train")

    tf.flags.DEFINE_integer(
        "check_every",
        default=105,
        help="How often (iterations) to check performances")

    tf.flags.DEFINE_integer(
        "save_model_every",
        default=20,
        help="How often (iterations) to save model")

    tf.flags.DEFINE_string(
        "model_dir",
        default="tensorboard/",
        help="Directory to put the model summaries, parameters and checkpoint.")


def select_one_random(list):
    """
    function to select one random item within the given list (used in parameter selection for random search)
    :param list:
    :return:
    """
    return list [randrange(len(list))]

def select_one_in_range(list, integer):
    """
    function to select a random value in the given range
    :param list:
    :param integer:
    :return:
    """

    rand = uniform(list[0], list[1])

    if integer:
        return round(rand)
    else:
        return rand

def shuffle_dev_set (train, validation):
    """
    function to shuffle train set and validation set
    :param train:
    :param validation:
    :return: new train and validation with shuffled item and keeping the same proportion of the orginal
    ones
    """
    dev_set = train + validation
    tot_len = len(dev_set)
    prop = float( len(train)  ) / float( len(train) + len(validation) )
    train_end = int (float(tot_len)*prop)
    shuffle(dev_set)
    return dev_set[:train_end] , dev_set[train_end:]

def shuffle_data(input,target):
    assert len(input) == len(target)
    perm = np.random.permutation([i for i in range(0,len(input))])
    input_shuffled = [input[i] for i in perm]
    target_shuffled = [target[i] for i in perm]
    return input_shuffled, target_shuffled

def max_arity (list):
    """
    funtion to get the msx_arity in data set
    :param list: list of tree(dataset)
    :return:
    """
    max_arity = 0
    for el in list:
        actual_arity = get_tree_arity(el)
        if actual_arity > max_arity:
            max_arity = actual_arity
    return max_arity


def get_tree_arity(t ):
    max_arity = len(t.children)
    for child in t.children:
        actual_arity = get_tree_arity(child)
        if actual_arity > max_arity:
            max_arity = actual_arity
    return max_arity


def get_max_arity(input_train, input_val, target_train, target_val):
    # compute max_arity
    train_image_max_arity = max_arity(input_train)
    val_image_max_arity = max_arity(input_val)
    image_max_arity = max(train_image_max_arity, val_image_max_arity)
    train_sen_max_arity = max_arity(target_train)
    val_sen_max_arity = max_arity(target_val)
    sen_max_arity = max(train_sen_max_arity, val_sen_max_arity)
    return image_max_arity, sen_max_arity
