import sys
sys.path.insert(0,'/home/serramazza/tf_tree')

from myCode.tree_defintions import *
from myCode.helper_functions import *
from myCode.validation import validation

def main():
    global embeddings
    global tag_dictionary

    #check arguments
    args = sys.argv[1:]
    if len(sys.argv)==0 or args[0] == "--help":
        help()
        exit(0)

    print ("begin")
    #########
    # Checkpoints and Summaries #TODO
    #########

    ##################
    # FLAGS
    ##################
    FLAGS = tf.flags.FLAGS
    define_flags()
    tf.enable_eager_execution()


    ###################
    # tree definition
    ###################
    image_tree = ImageTree()
    sentence_tree = SentenceTree()

    #load tree
    train_data, val_data = load_data(args)

    #get batch for traning
    input_train = get_image_batch(train_data)
    target_train = get_sentence_batch(train_data)
    input_val = get_image_batch(val_data)
    target_val = get_sentence_batch(val_data)


    #define parameters to search:
    parameters = []
    parameters.append([200,200]) #embedding_size
    parameters.append([ 100,100 ]) #max node count
    parameters.append([ 10,10 ]) #max_depth
    parameters.append([ 4,4 ]) #cut_arity
    parameters.append([0.05, 0.05]) #lambda
    parameters.append([0.005,0.005]) #beta
    parameters.append([ 0.3, 0.3]) #hidden_coefficient
    parameters.append([0.001, 0.001]) # learning
    parameters.append([0.02, 0.02]) #clip gradient
    parameters.append([6,6]) #batch size

    print ("begin experiments")
    validation(input_train,target_train,input_val,target_val,parameters,FLAGS,image_tree.tree_def,sentence_tree.tree_def,
               "parole_categoriche")


if __name__ == "__main__":
    main()
