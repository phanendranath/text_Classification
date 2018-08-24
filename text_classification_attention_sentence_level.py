import os
os.environ['KERAS_BACKEND'] = 'theano'
from data import load_preprocessed_data
from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.layers import Embedding, Input, Flatten
from keras.models import Model
from data_3d import load_preprocessed_data


SENT_LEN = 50
NUM_SENT = 20
EMBD_DIM = 100
INP_DIM = 22000


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1],)))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


if __name__ == "__main__":

    # This is two level, first sentence should be encoded,
    # then this encoded info should be used to further encode document
    # (made of sentences)

    # sentence encoder
    sentence_input = Input(shape=(SENT_LEN,))
    embeddings = Embedding(input_dim=INP_DIM, output_dim=EMBD_DIM,
                           input_length=SENT_LEN)(sentence_input)
    sentence_encoded = Bidirectional(LSTM(100, return_sequences=True))(embeddings)
    sentence_attention = AttLayer()(sentence_encoded)
    sentence_encoder_model = Model(inputs=sentence_input,
                                   outputs=sentence_attention)

    # Document encoder
    document_input = Input(shape=(NUM_SENT, SENT_LEN))
    sentences_encoded = TimeDistributed(sentence_encoder_model)(document_input)
    document_encoded = Bidirectional(LSTM(100, return_sequences=True))(sentences_encoded)
    document_attention = AttLayer()(document_encoded)

    fully_connected = Dense(32, activation="relu")(document_attention)
    outputs_ = Dense(1, activation="sigmoid")(fully_connected)

    hierarchial_model = Model(inputs=document_input, outputs=outputs_)
    hierarchial_model.summary()
    hierarchial_model.compile(loss="binary_crossentropy", optimizer="adam",
                              metrics=["accuracy"])

    reviews, labels = load_preprocessed_data(sent_len=SENT_LEN,
                                             num_sent=NUM_SENT)
    hierarchial_model.fit(x=reviews, y=labels, epochs=3, validation_split=0.2,
                          batch_size=32)

# ADAM GatedRU (100) + AttLayer was fed (20, 50). Out best config so far.
# Epoch 2/3
# 20000/20000 [==============================] - 1258s - loss: 0.1870 -
# acc: 0.9296 - val_loss: 0.2757 - val_acc: 0.8952
# Epoch 3/3
# 20000/20000 [==============================] - 1254s - loss: 0.0919 -
# acc: 0.9688 - val_loss: 0.3359 - val_acc: 0.8850

# Trying out the same as above config except with LSTM units this time
# Epoch 2/3
# 20000/20000 [==============================] - 1473s - loss: 0.2041 -
# acc: 0.9229 - val_loss: 0.2908 - val_acc: 0.8854
# Epoch 3/3
# 20000/20000 [==============================] - 1370s - loss: 0.1244 -
# acc: 0.9543 - val_loss: 0.3134 - val_acc: 0.8790
