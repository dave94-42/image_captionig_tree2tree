import xml.etree.ElementTree as ET
from myCode.tree_defintions import *
import tensorflow as tf
from tensorflow_trees.definition import Tree

def count_word_tag_occ(sen_tree : ET.Element ,  words_occ : list):

    value = sen_tree.attrib['value']
    #if current node is a word
    if sen_tree.tag=="leaf":
        words_occ.append(value)
    #if current word is a tag
    elif sen_tree.tag == "node":
        if (value not in shared_list.tags_idx) and (value!="ROOT"):
            shared_list.tags_idx.append(value)

    #recursion
    for child in sen_tree.getchildren():
        count_word_tag_occ(child, words_occ)


def read_tree_from_file(file):

    #open file
    tree = ET.parse(file)
    return tree.getroot()


def label_tree_with_real_data(xml_tree : ET.Element, final_tree : Tree,tokenizer):

    value = xml_tree.attrib["value"]
    if xml_tree.tag == "node" and value!="ROOT":
        #check if in frequent word in dev set otherwise label as others (last dimension)
        try:
            idx = shared_list.tags_idx.index(value)
        except:
            idx = len(shared_list.tags_idx)-1
        final_tree.node_type_id="POS_tag"
        final_tree.value=TagValue(representation=tf.one_hot(idx, len(shared_list.tags_idx)))
        final_tree.children = []
        for child in xml_tree.getchildren():
            final_tree.children.append(Tree(node_type_id="fake "))

    elif xml_tree.tag == "leaf":
        #check if in tag found in dev set otherwise label as others (last dimension)
        idx = tokenizer.texts_to_sequences([value])
        final_tree.node_type_id="word"
        final_tree.value=WordValue(representation=tf.one_hot(idx[0][0], WordValue.representation_shape))
        for child in xml_tree.getchildren():
            final_tree.children.append(Tree(node_type_id="fake "))


    #RECURSION
    elif xml_tree.tag == "node" and value=="ROOT":
        label_tree_with_real_data(xml_tree.getchildren()[0], final_tree,tokenizer)

    for child_xml, child_real in zip(xml_tree.getchildren(), final_tree.children):
        label_tree_with_real_data(child_xml, child_real,tokenizer)


def label_tree_with_sentenceTree(dev_data, tes_data, base_path):
    """
    function that given a tree (target for NN one) without sentence "label" it also with it
    :param dev_data:
    :param tes_data:
    :param base_path:
    :return:
    """
    #read xml file first
    for data in dev_data+tes_data:
        name = data['name']
        #after got file name, read tree from xml file
        tree = read_tree_from_file(base_path+name)
        data['sentence_tree'] = tree

    #count occurency of words
    word_occ = []
    for data in dev_data:
        count_word_tag_occ(data['sentence_tree'], word_occ)
    tokenizer,_ = extraxt_topK_words(word_occ,filters="~")
    TagValue.update_rep_shape(len(shared_list.tags_idx))

    #label tree with real data
    for data in dev_data+tes_data:
        final_tree = Tree(node_type_id="dummy root", children=[],value="dummy")
        label_tree_with_real_data(data['sentence_tree'], final_tree,tokenizer)
        final_tree = final_tree.children[0]
        if final_tree.value.abstract_value=="S":
            data['sentence_tree'] = final_tree
        else:
            idx = shared_list.tags_idx.index("S")
            tag=TagValue(representation=tf.one_hot(idx, len(shared_list.tags_idx)))
            S_node =Tree(node_type_id="POS_tag",children=[final_tree],value=tag)
            data['sentence_tree'] = S_node


def extraxt_topK_words(word_occ,filters):
    top_k = 50000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters='~')
    tokenizer.fit_on_texts(word_occ)
    # word number with 5 or more occurrebcy in training
    top_k = len((list(filter(lambda el: el[1] >= 5, tokenizer.word_counts.items()))))
    print(top_k)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters=filters)
    tokenizer.fit_on_texts(word_occ)
    tokenizer.word_index['<start>'] = 0
    tokenizer.index_word[0] = '<start>'
    shared_list.word_idx = tokenizer.word_index
    shared_list.idx_word = tokenizer.index_word
    WordValue.update_rep_shape(top_k)
    return tokenizer,top_k

