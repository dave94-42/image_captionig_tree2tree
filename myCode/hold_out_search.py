import sys

from myCode.tree_defintions import *
from myCode.helper_functions import *
from myCode.validation import validation
import os

def main():
    global embeddings
    global tag_dictionary

    # check arguments
    args = sys.argv[1:]
    if len(args) == 0 or args[0] == "--help":
        help()
        exit(0)

    print("begin")

    ##################
    # FLAGS
    ##################
    FLAGS = tf.flags.FLAGS
    define_flags()
    tf.enable_eager_execution()

    ###################
    # tree definition
    ###################
    tree_enoder = os.path.isfile(args[0])
    cnn_type = args[4].lower() if tree_enoder else None
    image_tree = ImageTree(cnn_type) if tree_enoder else None
    tree_decoder = os.path.isdir(args[2])
    sentence_tree = SentenceTree() if tree_decoder else None

    # load tree
    train_data, val_data, val_all_captions = load_data(args,tree_enoder,tree_decoder,cnn_type)
    # get batch for traning
    input_train,input_val = get_image_batch(train_data,val_data,image_tree==None)
    target_train, target_val = get_sentence_batch(train_data,val_data,tree_decoder,args[2])

    # define parameters to search:
    parameters = []
    parameters.append([300])  # embedding_size
    parameters.append([100])  # max node count
    parameters.append([10])  # max_depth
    parameters.append([4])  # cut_arity
    parameters.append([0.0003,0.001, 0.003])  # lambda #0.05,0.005,0.0005
    parameters.append([0.005, 0.01])  # beta
    parameters.append([0.3])  # hidden_coefficient
    parameters.append([0.001])  # learning
    parameters.append([0.02])  # clip gradient
    parameters.append([6,6])  # batch size
    parameters.append([200])  # word embedding #300
    parameters.append([1.0])  # hidden word size1.3

    #tree_decoder
    print("begin experiments")
    validation(input_train, target_train,input_val, target_val, parameters, FLAGS, image_tree,
               sentence_tree,name="tree_decoder_tutorial2",val_all_captions=val_all_captions)


if __name__ == "__main__":
    main()
