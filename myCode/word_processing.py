import tensorflow as tf
import numpy as np
import sys
import myCode.shared_POS_words_lists as shared_list

def get_ordered_nodes(embeddings, ops,TR, trees):
    """
    function to sort nodes and corresponding target words (before processing them)
    :param embeddings: all node embeddings
    :param ops:list of item to select among the wall node embeddings
    :return: inp (parents of node to generate)
            target(targets node list)
            batch_idx (map nodes to corresponding trees)
            perm2unsort (permutation to apply to have nodes in orginal order)
    """
    #take node numbs
    node_numb = [o.meta['node_numb'] for o in ops]

    if TR:
        #compute permutation to sort it and to "unsort" it
        perm2sort = np.argsort(node_numb)
        perm2unsort = inv(perm2sort)
        #take targets and check its length, sort them if we are in TR
        targets = [o.meta['target'].value.representation for o in ops]
        assert len(node_numb) == len(targets)
        targets = [targets[i] for i in perm2sort]
        #sort nodes, targets and get batch_idxs and input
        node_numb = [node_numb[i] for i in perm2sort]
        ops = [ops[i] for i in perm2sort]
        batch_idxs = [o.meta['batch_idx'] for o in ops]
        inp =tf.gather(embeddings, node_numb)
    else:
        #build list to return
        targets=[]
        batch_idxs = []
        leafs_ordered = []
        for tree in trees:
            leafs_ordered.append([])
        #get leafs ordered
        for tree in trees:
            get_ordered_leafs(tree,leafs_ordered)
        #build batch idx, inp and perm2unsort
        for i in range(0, len(leafs_ordered)):
            for el in leafs_ordered[i]:
                batch_idxs.append(i)
        nodes_to_take = [node_numb for l in leafs_ordered for node_numb in l]
        inp = tf.gather(embeddings,nodes_to_take)
        perm2unsort = np.argsort(nodes_to_take)

    #assert
    assert len(node_numb) == inp.shape[0]
    assert len(node_numb) == len(batch_idxs)

    return inp,targets,batch_idxs,perm2unsort

#########################
#functions used in words prediction
#######################
def words_predictions(word_module, batch_idxs, inp, targets, TR,roots_emb,
                      root_only_in_fist_LSTM_time,perm2unsort):

    #take sentences length
    sentences_len = get_sentences_length(batch_idxs,TR)
    #prepare data (reshape as expected)
    inputs, targets_padded_sentences = zip_data(inp, sentences_len, targets,TR, root_only_in_fist_LSTM_time)
    if TR:
        #if training or teacher forcing
        predictions = teacher_forcing(word_module,inputs, targets_padded_sentences,roots_emb,
                                      root_only_in_fist_LSTM_time)
    else:
        #otherwise sampling
        predictions = sampling(word_module,inputs, roots_emb,root_only_in_fist_LSTM_time)
    #unzip data (reshape as 2D matrix)
    vals = unzip_data(predictions,sentences_len,perm2unsort)
    return vals

#TODO handle root in each time stamp
def zip_data(inp, sentences_len, targets,TR, root_only_in_fist_LSTM_time):
    """
    function to get data in format expected by RNN i.e. (n_senteces)*(sen_max_lenght)*(representation_size)
    :param inp:  input as 2D matrix
    :param sentences_len:  list of sentences length
    :param targets: list of targets nodes
    :param TR:  whether we are in Traning or not
    :param root_only_in_fist_LSTM_time: whether to use tree toot only in the first timestamp or not
    :return:
    """
    max_len = np.amax(sentences_len)
    padded_input = None
    targets_padded_sentences = None
    current_node = 0
    current_tree=0
    # reshape as (n_senteces)*(sen_max_lenght)*(representation_size)
    for el in sentences_len:
        # first input part

        # take nodes belonging to current sentence, pad them to max sentences length and concatenate them all together
        current_sen =inp[current_node:current_node+el]
        if not root_only_in_fist_LSTM_time:
            sys.exit("to implement")

        padding = tf.constant([[0, (max_len - el)], [0, 0]])
        current_sen = tf.pad(current_sen, padding, 'CONSTANT')
        current_sen = tf.expand_dims(current_sen,axis=0)
        padded_input = update_matrix(current_sen, padded_input,ax=0)
        if TR:
            # targets only if available i.e. if we are in TR
            current_target = []
            for item in targets[current_node:(current_node + el)]:
                # take as target all words in current sentence except the last one (it will be never use as target)
                target_reshaped = tf.expand_dims(tf.expand_dims(item,axis=0),axis=0)
                current_target.append(target_reshaped)
            # as before pad the sentence to max length, then concatenate with other ones
            current_target = tf.concat([item for item in current_target], axis=1)
            padding = tf.constant([[0, 0], [0, (max_len - el)], [0, 0]])
            current_target = tf.pad(current_target, padding, 'CONSTANT')
            targets_padded_sentences = update_matrix(current_target,targets_padded_sentences,ax=0)
        #in any case update current node pointer
        current_node+=el
        current_tree+=1
    assert current_node == np.sum(sentences_len)
    return padded_input, targets_padded_sentences


def unzip_data(predictions,sentences_len,perm2unsort):
    """
    function to unzip rnn result i.e. go back in representation as 2D matrix and go back to previous order of nodes
    :param predictions: predictions generated by word module
    :param sentences_len: lengths of th sentences
    :param perm2unsort:  permutation to apply in order to get the excpected order
    :return:
    """
    vals = None
    for i in range (len(sentences_len)):
        current_sen_padded =predictions[i]
        current_sen = current_sen_padded[0:sentences_len[i]]
        vals = update_matrix(current_sen,vals,ax=0)
    vals = tf.gather(vals, perm2unsort)
    assert np.sum(sentences_len) == vals.shape[0]
    return vals

def teacher_forcing(word_module, inputs, targets_padded_sentences,roots):

    #predictions = teacher_forcing_NIC(word_module, inputs, rnn, roots, roots_conc_mode,
    #                                  targets_padded_sentences)

    hidden = word_module.reset_state(batch_size=inputs.shape[0])
    dec_input = tf.expand_dims([shared_list.word_idx['<start>']] * inputs.shape[0], 1)
    predictions = None
    for i in range(inputs.shape[1]):
        current_pred,hidden = word_module.call(dec_input,roots,hidden,parents = tf.expand_dims (inputs[:,i,:],axis=1))
        dec_input = tf.expand_dims( tf.argmax(targets_padded_sentences[:,i,:],axis=-1) , axis=1)
        predictions = update_matrix(tf.expand_dims(current_pred,axis=1),predictions,ax=1)
    return predictions



def sampling(word_module, inputs, roots):

    sentences = word_module.sampling(roots,shared_list.word_idx,shared_list.idx_word,inputs.shape[1],parents=inputs)
    #sentences = sampling_NIC(embedding, final_layer, inputs, rnn, roots, roots_conc_mode)
    return sentences



######################
#NIC code
######################
def teacher_forcing_NIC(embedding_layer, final_layer, inputs, rnn, roots, roots_conc_mode, targets_padded_sentences):
    embeddings = embedding_layer(tf.argmax(targets_padded_sentences, axis=2))
    if roots_conc_mode:
        assert (roots.shape[-1] == embeddings.shape[-1]), "embedding dimensions must be the same"
        roots = tf.expand_dims(roots, axis=1)
        embeddings = tf.concat([roots, embeddings], axis=1)
        padding = tf.constant([[0, 0], [1, 0], [0, 0]])
        inputs = tf.pad(inputs, padding)
    else:
        sys.exit("to implement")
    rnn_input = tf.concat([inputs, embeddings], axis=2)
    rnn_out, state_h, state_c = rnn(rnn_input)
    predictions = final_layer(rnn_out)
    return predictions

def sampling_NIC(embedding, final_layer, inputs, rnn, roots, roots_conc_mode):
    sentences = None
    # concatenate in the proper way, parent embedding and root embedding
    if roots_conc_mode != None:
        last_analized_words = tf.expand_dims(roots, axis=1)
    else:
        sys.exit("to implement")
    max_sentences_len = inputs.shape[1]
    for i in range(0, max_sentences_len):
        # fed the i-th input to the rnn, save its status and upadate embeggins to use at iteration (i+1)-th
        current_input = inputs[:, i, :]
        current_input = tf.expand_dims(current_input, axis=1)
        current_input = tf.concat([current_input, last_analized_words], axis=2)
        if i == 0:
            rnn_out, state_h, state_c = rnn(current_input)
            state = [state_h, state_c]
        else:
            rnn_out, state_h, state_c = rnn(current_input, initial_state=state)
            state = [state_h, state_c]
        predictions = final_layer(rnn_out)
        sentences = update_matrix(predictions, sentences, ax=1)
        last_analized_words = embedding(tf.argmax(predictions, axis=2))
    return sentences

#######################
#helper functions
######################

def update_matrix(current, total, ax):
    """
    function that concatenate current matrix in total matrix,
    :param current: tensor to add
    :param total: final tensor to return
    :param ax: axis to concatenate
    :return:
    """
    if total == None:
        total = current
    else:
        total = tf.concat([total, current], axis=ax)
    return total

def inv(perm):
    """
    function returning inverse of given permutation
    :param perm:  permutation to invert
    :return:
    """
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def get_sentences_length(batch_idxs,TR):

    if TR:
        current_tree = 0
        sentences_len = []
        current_len = 0
        for el in batch_idxs:
            if el == current_tree:
                current_len += 1
            else:
                current_tree += 1
                sentences_len.append(current_len)
                current_len = 1
        sentences_len.append(current_len)
    else:
        sentences_len = []
        n_sentences = np.max(batch_idxs)+1
        for i in range(n_sentences):
            sentences_len.append(batch_idxs.count(i))
    assert np.sum(sentences_len) == len(batch_idxs)
    return sentences_len

def get_ordered_leafs(tree, l : list):
    """
    function to get ordered leafs from left to right
    :param tree: tree to visit
    :param l: list in which append node number
    :return:
    """
    if tree.node_type_id=="word":
        l[tree.meta['batch_idx']].append(tree.meta['node_numb'])
    for c in tree.children:
        get_ordered_leafs(c,l)
