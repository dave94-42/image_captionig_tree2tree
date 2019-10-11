from tensorflow_trees.definition import TreeDefinition, NodeDefinition
import tensorflow as tf
import myCode.shared_POS_words_lists as shared_list


###########
#image tree
###########
class ImageValue(NodeDefinition.Value):
    """
    class modelling single tree image node
    """
    representation_shape = 384 #this shape come from alexNet
    class_value = False

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        return t.numpy()

    @staticmethod
    def abstract_to_representation_batch(v):
        return tf.Variable(v,dtype=tf.float32)


class ImageTree:
    """
    class modelling whole image tree
    """
    def __init__(self):

        self.tree_def = TreeDefinition(node_types=[
            NodeDefinition("othersInternal",may_root=True,arity=NodeDefinition.VariableArity(min_value=5),value_type=ImageValue),
            NodeDefinition("doubleInternal",may_root=True,arity=NodeDefinition.FixedArity(4),value_type=ImageValue),
            NodeDefinition("internal",may_root=True,arity=NodeDefinition.FixedArity(2),value_type=ImageValue),
            NodeDefinition("leaf",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=ImageValue)
        ])

        self.node_types = self.tree_def.node_types



#############
#sentence tree
#############


class TagValue(NodeDefinition.Value):
    """
    class modelling POS tag vale as one hot encoding
    """
    representation_shape = 0 #n* of different pos tag in flick8k train set
    class_value = True

    @staticmethod
    def update_rep_shape(shape):
        TagValue.representation_shape = shape

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        idx = tf.math.argmax(t[0])
        try:
            ris = shared_list.tags_idx[idx]
        except IndexError:
            ris = "not_found"
        return ris

    @staticmethod
    def abstract_to_representation_batch(v):
        """
        return the associated value to the key v(argument of the function)
        :param v:
        :return:
        """
        if type(v)==list:
            ris=[]
            for el in v:
                idx = shared_list.tags_idx.index(el)
                ris.append( tf.one_hot(idx, TagValue.representation_shape ) )
            return ris
        else:
            idx = shared_list.tags_idx.index(v)
            return  tf.one_hot(idx,TagValue.representation_shape)

class WordValue(NodeDefinition.Value):
    """
    class modelling word value i.e. emebedding vector
    """
    representation_shape = 0 #dimension of embedding currently used
    class_value = True

    @staticmethod
    def update_rep_shape(shape):
        WordValue.representation_shape = shape

    @staticmethod
    def representation_to_abstract_batch(t:tf.Tensor):
        idx = tf.math.argmax(t[0])
        try:
            ris = shared_list.word_idx[idx]
        except IndexError:
            ris = "not_found"
        return ris

    @staticmethod
    def abstract_to_representation_batch(v):
        """
        return the associated value to the key v(argument of the function)
        :param v:
        :return:
        """
        if type(v)==list:
            ris=[]
            for el in v:
                idx = shared_list.word_idx.index(el)
                ris.append( tf.one_hot(idx,WordValue.representation_shape) )
            return ris
        else:
            idx = shared_list.word_idx.index(v)
            return tf.one_hot(idx,WordValue.representation_shape)

class SentenceTree:
    """
    class representing sentence tree.
    """
    def __init__(self):
        self.tree_def = TreeDefinition(node_types=[
            NodeDefinition("POS_tag",may_root=True,arity=NodeDefinition.VariableArity(min_value=1),value_type=TagValue),
            NodeDefinition("word",may_root=False,arity=NodeDefinition.FixedArity(0),value_type=WordValue)
        ])

        self.node_types = self.tree_def.node_types
