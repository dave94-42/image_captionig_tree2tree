import xml.etree.ElementTree as ET
from myCode.tree_defintions import *
import tensorflow as tf
from tensorflow_trees.definition import Tree

def count_word_tag_occ(sen_tree : ET.Element ,  word_dict : dict):
    """
    function that count word occurency in dataset
    :param sen_tree:
    :param word_dict:
    :return:
    """
    value = sen_tree.attrib['value']
    #if current node is a word
    if sen_tree.tag=="leaf":
        if value.lower() in word_dict.keys():
            word_dict[value.lower()] = word_dict[value.lower()] +1
        else:
            word_dict[value.lower()] = 1
    #if current word is a tag
    elif sen_tree.tag == "node":
        if (value not in shared_list.tags_idx) and (value!="ROOT"):
            shared_list.tags_idx.append(value)

    #recursion
    for child in sen_tree.getchildren():
        count_word_tag_occ(child, word_dict)


def read_tree_from_file(file):
    """
    function that read parse tree from xml file
    :param file:
    :param embeddings:
    :param dictionary:
    :param name:
    :return:
    """
    #open file
    tree = ET.parse(file)
    return tree.getroot()


def label_tree_with_real_data(xml_tree : ET.Element, final_tree : Tree):
    """
    function that given tree as read form xml file, "label" current tree with tree data for NN
    :param data:
    :param final_tree:
    :return:
    """
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
        try:
            idx = shared_list.word_idx.index(value.lower())
        except:
            idx = len(shared_list.word_idx)-1
        final_tree.node_type_id="word"
        final_tree.value=WordValue(representation=tf.one_hot(idx, len(shared_list.word_idx)))
        for child in xml_tree.getchildren():
            final_tree.children.append(Tree(node_type_id="fake "))


    #RECURSION
    elif xml_tree.tag == "node" and value=="ROOT":
        label_tree_with_real_data(xml_tree.getchildren()[0], final_tree)

    for child_xml, child_real in zip(xml_tree.getchildren(), final_tree.children):
        label_tree_with_real_data(child_xml, child_real)


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
        #after got fila name, read tree from xml file
        tree = read_tree_from_file(base_path+name)
        data['sentence_tree'] = tree

    #count occurency of words
    word_occ = {}
    for data in dev_data:
        count_word_tag_occ(data['sentence_tree'], word_occ)

    #filter dict to contains only words with 5 or more occurrency
    shared_list.word_idx = [ key for (key,value) in word_occ.items() if value >= 5]
    #add to them unknow word/tag and count their occurency for demension shape
    shared_list.word_idx.append("not_found")
    shared_list.tags_idx.append("not_found")
    WordValue.update_rep_shape(len(shared_list.word_idx))
    TagValue.update_rep_shape(len(shared_list.tags_idx))

    #label tree with real data
    for data in dev_data+tes_data:
        final_tree = Tree(node_type_id="dummy root", children=[],value="dummy")
        label_tree_with_real_data(data['sentence_tree'], final_tree)
        final_tree = final_tree.children[0]
        if final_tree.value.abstract_value=="S":
            data['sentence_tree'] = final_tree
        else:
            idx = shared_list.tags_idx.index("S")
            tag=TagValue(representation=tf.one_hot(idx, len(shared_list.tags_idx)))
            S_node =Tree(node_type_id="POS_tag",children=[final_tree],value=tag)
            data['sentence_tree'] = S_node



"""
def read_tree_from_file(file,embeddings,dictionary,name):

    #open file
    tree = ET.parse(file)
    root = tree.getroot()
    #dummy root
    dummy = Tree(node_type_id="dummy",children=[],value="dummy")
    #get tree really read the tree
    get_tree(dummy, root, dictionary, embeddings,name)
    #return child of dummy root i.e. the real root
    return dummy.children[0]
"""
