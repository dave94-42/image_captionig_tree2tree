import tensorflow.contrib.eager as tfe
import tensorflow.contrib.summary as tfs
from myCode.helper_functions import max_arity,shuffle_data
from myCode.models import *
from  myCode.helper_functions import select_one_in_range
from tensorflow_trees.definition import Tree
from myCode.load_model import restore_model,predict_test_dataSet
import shutil

def train_model(FLAGS, decoder, encoder, input_train, input_val, target_train, target_val,
                optimizer, beta,lamb,clipping,batch_size,n_exp, name, final=False, test=None):

    best_n_it = 0
    best_loss = 100

    checkpoint_prefix =  "/home/davide/workspace/tesi/tf_tree/saved_while_training/save.ckpt"

    if final:
        FLAGS.max_iter = 2000
        FLAGS.check_every = 10
        checkpoint_prefix =  "/home/davide/workspace/tesi/tf_tree/saved_model_lungo_shuffled2/save.ckpt"

    #tensorboard
    summary_writer = tfs.create_file_writer(FLAGS.model_dir+name+"/" +str(n_exp), flush_millis=1000)
    summary_writer.set_as_default()

    with tfs.always_record_summaries():
        for i in range(FLAGS.max_iter):
            loss_struct=0
            loss_value=0
            loss_POS = 0
            loss_word = 0

            #shuffle dataset at beginning of each iteration
            input_train,target_train = shuffle_data(input_train,target_train)

            for j in range(0,len(input_train),batch_size):
                with tfe.GradientTape() as tape:

                    current_batch_input=input_train[j:j+batch_size]
                    current_batch_target = target_train[j:j+batch_size]

                    # encode and decode datas
                    batch_enc = encoder(current_batch_input)
                    root_emb=batch_enc.get_root_embeddings()
                    batch_dec = decoder(encodings=root_emb, targets=current_batch_target)

                    # compute global loss
                    loss_struct_miniBatch, loss_values_miniBatch = batch_dec.reconstruction_loss()
                    loss_value__miniBatch = loss_values_miniBatch["POS_tag"] + loss_values_miniBatch["word"]
                    loss_miniBatch = loss_value__miniBatch+loss_struct_miniBatch

                    #compute minibatch loss
                    loss_struct += loss_struct_miniBatch
                    loss_value += loss_value__miniBatch
                    loss_POS += loss_values_miniBatch["POS_tag"]
                    loss_word +=  loss_values_miniBatch["word"]

                    variables = encoder.variables + decoder.variables
                    weights = encoder.weights + decoder.weights

                    #compute h and w norm for refularization
                    h_norm= tf.norm(root_emb)
                    w_norm=0
                    for w in weights:
                        norm = tf.norm(w)
                        if norm >= 0.001:
                            w_norm += norm

                    # compute gradient
                    grad = tape.gradient(loss_miniBatch+ beta*w_norm +lamb*h_norm, variables)
                    gnorm = tf.global_norm(grad)
                    grad, _ = tf.clip_by_global_norm(grad, clipping, gnorm)
                    tfs.scalar("norms/grad", gnorm)
                    tfs.scalar("norms/h_norm", h_norm)
                    tfs.scalar("norms/w_norm", w_norm)

                    # apply optimizer on gradient
                    optimizer.apply_gradients(zip(grad, variables), global_step=tf.train.get_or_create_global_step())

            loss_struct /= (int(len(input_train)/batch_size)+1)
            loss_value /= (int(len(input_train)/batch_size)+1)
            loss_POS  /= (int(len(input_train)/batch_size)+1)
            loss_word /= (int(len(input_train)/batch_size)+1)
            loss = loss_struct+loss_value

            print("loss_POS ", loss_POS.numpy(), " loss_word ", loss_word.numpy())

            tfs.scalar("loss/loss_struc", loss_struct)
            tfs.scalar("loss/loss_value", loss_value)
            tfs.scalar("loss/loss_value_POS", loss_POS)
            tfs.scalar("loss/loss_value_word", loss_word)


        # print stats
            if i % FLAGS.check_every == 0:

                var_to_save = encoder.variables+encoder.weights + decoder.variables+decoder.weights + optimizer.variables()
                tfe.Saver(var_to_save).save(checkpoint_prefix,global_step=tf.train.get_or_create_global_step())
                if final:
                    predict_test_dataSet(decoder,encoder,input_val,target_val,test,
                        "/home/davide/workspace/tesi/tf_tree/pred_model/lungo_shuffled/pred"+str(i)+".json")

                # get validation loss
                batch_val_enc = encoder(input_val)
                batch_val_dec = decoder(encodings=batch_val_enc.get_root_embeddings(), targets=target_val)
                loss_struct_val, loss_values_validation = batch_val_dec.reconstruction_loss()
                loss_validation = loss_struct_val + loss_values_validation["POS_tag"]+loss_values_validation["word"]

                print("iteration ", i, " loss train value is ", loss_word, " loss train POS is ", loss_POS , "\n",
                      " loss validation word is ", loss_values_validation["word"], " loss validation POS is ", loss_values_validation["POS_tag"])
                tfs.scalar("overlaps/struct_validation", loss_struct_val)
                tfs.scalar("overlaps/value_validation_POS", loss_values_validation["POS_tag"])
                tfs.scalar("overlaps/value_validation_word", loss_values_validation["word"])
                tfs.scalar("overlaps/loss_validation", loss_validation)

                #early stopping
                if not final:
                    if best_loss > loss_values_validation["word"]:
                        best_loss = loss_values_validation["word"]
                        best_n_it = i
                    else:
                        print("restoring previous model")
                        restore_model(encoder,decoder,"/home/davide/workspace/tesi/tf_tree/saved_while_training/")
                        break

        print("evalutaing it")
        matched_word_sup, matched_word_uns, matched_pos_sup,matched_pos_uns = evaluate_perfoemance(
            batch_val_dec, batch_val_enc, decoder,target_val)

    return matched_word_sup, matched_word_uns, matched_pos_sup,matched_pos_uns,best_n_it


def evaluate_perfoemance(batch_val_dec, batch_val_enc, decoder, target_val):

    batch_unsuperv = decoder(encodings=batch_val_enc.get_root_embeddings())

#overlaps_s_avg, overlaps_v_avg,np.sum(tot_pos),np.sum(matched_pos),np.sum(tot_word),np.sum(matched_word)
    _, v_avg_sup, tot_pos_sup, matched_pos_sup, total_word_sup ,matched_word_sup = Tree.compare_trees(target_val, batch_val_dec.decoded_trees)
    s_avg, v_avg, tot_pos_uns, matched_pos_uns, total_word_uns ,matched_word_uns= Tree.compare_trees(target_val, batch_unsuperv.decoded_trees)

    tfs.scalar("overlaps/supervised/value_avg", v_avg_sup)
    tfs.scalar("overlaps/supervised/total_POS", tot_pos_sup)
    tfs.scalar("overlaps/supervised/matched_POS", matched_pos_sup)
    tfs.scalar("overlaps/supervised/total_words", total_word_sup)
    tfs.scalar("overlaps/supervised/matched_words", matched_word_sup)
    tfs.scalar("overlaps/unsupervised/struct_avg", s_avg)
    tfs.scalar("overlaps/unsupervised/value_avg", v_avg)
    tfs.scalar("overlaps/unsupervised/total_POS", tot_pos_uns)
    tfs.scalar("overlaps/unsupervised/matched_POS", matched_pos_uns)
    tfs.scalar("overlaps/unsupervised/total_words", total_word_uns)
    tfs.scalar("overlaps/unsupervised/matched_words", matched_word_uns)

    return matched_word_sup, matched_word_uns, matched_pos_sup,matched_pos_uns


def validation(input_train, target_train, input_val, target_val ,parameters, FLAGS,input_tree, target_tree, name: str) :

    #open file
    f= open(name+".txt","ab", buffering=0)

    #compute max_arity
    train_image_max_arity = max_arity(input_train)
    val_image_max_arity = max_arity(input_val)
    image_max_arity = max(train_image_max_arity,val_image_max_arity)

    train_sen_max_arity = max_arity(target_train)
    val_sen_max_arity = max_arity(target_val)
    sen_max_arity = max(train_sen_max_arity,val_sen_max_arity)

    #selected actual parameter to try
    for i in range (0,20):
        embedding_size = select_one_in_range(parameters[0],integer=True)
        max_node_count = select_one_in_range(parameters[1],integer=True)
        max_depth = select_one_in_range(parameters[2],integer=True)
        cut_arity = select_one_in_range(parameters[3],integer=True)
        lamb = select_one_in_range(parameters[4],integer=False)
        beta = select_one_in_range(parameters[5],integer=False)
        hidden_coeff = select_one_in_range(parameters[6],integer=False)
        learning_rate = select_one_in_range(parameters[7],integer=False)
        clipping = select_one_in_range(parameters[8],integer=False)
        batch_size = select_one_in_range(parameters[9],integer=True)
        batch_size = pow(2,batch_size)

        activation = getattr(tf.nn, FLAGS.activation)

        decoder, encoder = get_encoder_decoder(emb_size=embedding_size,cut_arity=cut_arity,max_arity=max(image_max_arity,
            sen_max_arity),max_node_count=max_node_count,max_depth=max_depth,hidden_coeff=hidden_coeff,
            activation=activation,image_tree=input_tree,sentence_tree=target_tree)

        #train
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        matched_word_sup, matched_word_uns, matched_pos_sup,matched_pos_uns,best_n_it = train_model(FLAGS=FLAGS,decoder=decoder,encoder=encoder,
            input_train=input_train, input_val=input_val,target_train=target_train,target_val=target_val,optimizer=optimizer,
            beta=beta,lamb=lamb,clipping=clipping,batch_size=batch_size,n_exp=i,name=name,final=False,test=None)


        string = "\n" +str(i) +")models with parameters emb_size " + str (embedding_size) + " max node count " + str(max_node_count) + \
                 " max_depth " + str(max_depth) + " cut arity " + str(cut_arity) + \
                 " lamdda " + str(lamb) + " beta " + str(beta) + \
                 " hidden coeff " + str(hidden_coeff) +" learn rate " + str(learning_rate) + " clipping "+ str(clipping) + \
                 " batch size " + str(batch_size) + " has riched matched matched word sup " + str(matched_word_sup)+ \
                 " ,matched POS supervised " + str(matched_pos_sup) + " ,matched word unsupervised " + str(matched_word_uns)  +\
                 " ,matched POS unsupervised " + str(matched_pos_uns) +  " and struct accuracy " + \
                 " in "+ str(best_n_it) + " itertions\n"

        f.write(str.encode(string))
        shutil.rmtree("/home/serramazza/tf_tree/saved_while_training/")
        print ("experiment " + str(i) + " out of 20 finished\n")
