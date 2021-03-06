from myCode.CNN_encoder import CNN_Encoder
from myCode.RNN_decoder import BahdanauAttention, RNN_Decoder
from myCode.tree_defintions import WordValue
from tensorflow_trees.decoder import Decoder, DecoderCellsBuilder
from tensorflow_trees.encoder import Encoder, EncoderCellsBuilder


def get_encoder_decoder(emb_tree_size, cut_arity, hidden_word,max_arity, max_node_count, max_depth, hidden_coeff,
                        activation,emb_word_size,image_tree, sentence_tree):

    if image_tree==None:
        encoder = CNN_Encoder(emb_tree_size)
    else:
        encoder = Encoder(tree_def=image_tree.tree_def, embedding_size=emb_tree_size, cut_arity=cut_arity, max_arity=max_arity,
                          variable_arity_strategy="FLAT",name="encoder",

                cellsbuilder=EncoderCellsBuilder(EncoderCellsBuilder.simple_cell_builder(
                    hidden_coef=hidden_coeff, activation=activation,gate=True),

                EncoderCellsBuilder.simple_dense_embedder_builder(activation=activation)))

    WordValue.set_embedding_size(emb_word_size)

    if sentence_tree==None:
        decoder = RNN_Decoder(emb_word_size,hidden_word,WordValue.representation_shape)
    else:
        attention = BahdanauAttention(hidden_word) if image_tree==None else None
        decoder = Decoder(tree_def=sentence_tree.tree_def, embedding_size=emb_tree_size, max_arity=max_arity,max_depth=max_depth,
                          max_node_count=max_node_count, cut_arity=cut_arity, variable_arity_strategy="FLAT",
                hidden_word=hidden_word, attention=attention,

                cellsbuilder=DecoderCellsBuilder(distrib_builder=DecoderCellsBuilder.simple_distrib_cell_builder(
                    hidden_coef=hidden_coeff,activation=activation),

                    categorical_value_inflater_builder=DecoderCellsBuilder.simple_1ofk_value_inflater_builder(
                    hidden_coef=hidden_coeff,activation=activation),

                    dense_value_inflater_builder=None,

                    node_inflater_builder=DecoderCellsBuilder.simple_node_inflater_builder(hidden_coef=hidden_coeff,
                    activation=activation,gate=True)))

    return decoder, encoder


