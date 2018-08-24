from keras.layers import GRU, Bidirectional, Dense, TimeDistributed
from keras.layers import Embedding, Input, Flatten
from keras.models import Model
from data_3d import load_preprocessed_data


SENT_LEN = 50
NUM_SENT = 20
EMBD_DIM = 100
INP_DIM = 22000


if __name__ == "__main__":

    # This is two level, first sentence should be encoded,
    # then this encoded info should be used to further encode document
    # (made of sentences)

    # sentence encoder
    sentence_input = Input(shape=(SENT_LEN,))
    embeddings = Embedding(input_dim=INP_DIM, output_dim=EMBD_DIM,
                           input_length=SENT_LEN)(sentence_input)
    sentence_encoded = Bidirectional(GRU(100))(embeddings)
    sentence_encoder_model = Model(inputs=sentence_input,
                                   outputs=sentence_encoded)

    # Document encoder
    document_input = Input(shape=(NUM_SENT, SENT_LEN))
    sentences_encoded = TimeDistributed(sentence_encoder_model)(document_input)
    document_encoded = Bidirectional(GRU(100))(sentences_encoded)

    fully_connected = Dense(32, activation="relu")(document_encoded)
    outputs_ = Dense(1, activation="sigmoid")(fully_connected)

    hierarchial_model = Model(inputs=document_input, outputs=outputs_)
    hierarchial_model.summary()
    hierarchial_model.compile(loss="binary_crossentropy", optimizer="adam",
                              metrics=["accuracy"])

    reviews, labels = load_preprocessed_data(sent_len=SENT_LEN,
                                             num_sent=NUM_SENT)
    hierarchial_model.fit(x=reviews, y=labels, epochs=3, validation_split=0.2)

# Analysis on dataset revealed average number of sentences per review is 14
# ADAM with only 64 LSTM cells and input of (5, 100)
# Epoch 2/3
# 20000/20000 [==============================] - 375s - loss: 0.2540
#  - acc: 0.8993 - val_loss: 0.4173 - val_acc: 0.8180
# Epoch 3/3
# 20000/20000 [==============================] - 363s - loss: 0.1402
#  - acc: 0.9482 - val_loss: 0.5134 - val_acc: 0.8156

# ADAM with only 64 LSTM cells and input (20, 50)
# Epoch 2/3
# 20000/20000 [==============================] - 739s - loss: 0.1829 -
# acc: 0.9338 - val_loss: 0.2939 - val_acc: 0.8838
# Epoch 3/3
# 20000/20000 [==============================] - 746s - loss: 0.0996 -
# acc: 0.9668 - val_loss: 0.3442 - val_acc: 0.8796

# ADAM with 100 LSTM cells and input (20, 50)
# Epoch 2/3
# 20000/20000 [==============================] - 1233s - loss: 0.1909 -
#  acc: 0.9289 - val_loss: 0.3133 - val_acc: 0.8804
# Epoch 3/3
# 20000/20000 [==============================] - 1225s - loss: 0.0948 -
#  acc: 0.9670 - val_loss: 0.3576 - val_acc: 0.8828

# ADAM with 300 LSTM cells and input (20, 20) Trainable params: 5,344,065
# Epoch 2/3
# 20000/20000 [==============================] - 1604s - loss: 0.2029 -
# acc: 0.9227 - val_loss: 0.3085 - val_acc: 0.8712
# Epoch 3/3
# 20000/20000 [==============================] - 1526s - loss: 0.1091 -
# acc: 0.9623 - val_loss: 0.3854 - val_acc: 0.8662

# ADAM with 100 GRU cells and input (20, 50)
# Epoch 2/3
# 20000/20000 [==============================] - 871s - loss: 0.1702 -
# acc: 0.9355 - val_loss: 0.2828 - val_acc: 0.8862
# Epoch 3/3
# 20000/20000 [==============================] - 854s - loss: 0.0754 -
# acc: 0.9742 - val_loss: 0.3748 - val_acc: 0.8816
