import tensorflow as tf
from myCode.word_processing import update_matrix

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden,parents=None):
        # defining attention as a separate model
        context_vector = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        ctx_vec = tf.expand_dims(context_vector, 1)
        x = tf.concat([ctx_vec, x], axis=-1) if parents==None else tf.concat([ctx_vec,x,parents],axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def sampling(self,features,wi,iw,max_length,parents=None):

        hidden = self.reset_state(batch_size=features.shape[0])
        dec_input = tf.expand_dims([wi['<start>']] * features.shape[0], 1)
        to_return=None
        #prepare result array
        result =[]
        for i in range(int(features.shape[0])):
            result.append([])

        #sampling i-th words of the different captions to generate in parallel
        for i in range(max_length):
            current_parents = tf.expand_dims (parents[:,i,:],axis=1) if parents!=None else None
            predictions, hidden = self.call(dec_input, features, hidden,parents=current_parents)
            predicted_ids = tf.squeeze( tf.random.multinomial(predictions, 1), axis=1).numpy()
            #predicted_id = tf.argmax(predictions,axis=-1).numpy()[0]
            dec_input = tf.expand_dims(predicted_ids, 1)

            if parents==None:
                for j in range(len(predicted_ids)):
                    result[j].append(iw[predicted_ids[j]])
            else:
                to_return=update_matrix(tf.expand_dims(predictions,axis=1),to_return,ax=1)

        if parents==None:
            #extract pred sentence up to <end> token
            to_return = []
            for el in result:
                try:
                    end = el.index('<end>')
                    to_return.append(el[:end])
                except ValueError:
                    to_return.append(el)

        return to_return
