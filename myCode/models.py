from tensorflow_trees.decoder import Decoder, DecoderCellsBuilder
from tensorflow_trees.encoder import Encoder, EncoderCellsBuilder
from myCode.tree_defintions import WordValue

# cut_arity è il numero di nodi che vengono passati direttamente all'encodder flat, se = n l'input n+1 è
# l'attention applicata a tutti gli altri input
# variable_arity_strategy se usare flat o altro, forse unica scelta possibile è flat???
# cell.hidden_coeff regola la dimensione dell'hidden layer, in particolare dim(hidden) = dim(input+output ) ∗ h_coeff,
# cell.gate è una forma di attenzione, calcola un coefficiente utlizzato in una comb. lineare? per ogni sotto albero
# questo coeff viene utilizzato come peso nella combinazione lineare



def get_encoder_decoder(emb_tree_size, cut_arity, hidden_word,max_arity, max_node_count, max_depth, hidden_coeff,
                        activation,emb_word_size,image_tree, sentence_tree):

    encoder = Encoder(tree_def=image_tree, embedding_size=emb_tree_size, cut_arity=cut_arity, max_arity=max_arity,
                      variable_arity_strategy="FLAT",name="encoder",

            cellsbuilder=EncoderCellsBuilder(EncoderCellsBuilder.simple_cell_builder(
                hidden_coef=hidden_coeff, activation=activation,gate=True),

            EncoderCellsBuilder.simple_dense_embedder_builder(activation=activation)))

    WordValue.set_embedding_size(emb_word_size)

    decoder = Decoder(tree_def=sentence_tree, embedding_size=emb_tree_size, max_arity=max_arity,max_depth=max_depth,
                      max_node_count=max_node_count, cut_arity=cut_arity, variable_arity_strategy="FLAT",
            hidden_word=hidden_word,

            cellsbuilder=DecoderCellsBuilder(distrib_builder=DecoderCellsBuilder.simple_distrib_cell_builder(
                hidden_coef=hidden_coeff,activation=activation),

                categorical_value_inflater_builder=DecoderCellsBuilder.simple_1ofk_value_inflater_builder(
                hidden_coef=hidden_coeff,activation=activation),

                dense_value_inflater_builder=None,

                node_inflater_builder=DecoderCellsBuilder.simple_node_inflater_builder(hidden_coef=hidden_coeff,
                activation=activation,gate=True)))



    return decoder, encoder

