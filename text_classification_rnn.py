from data import load_preprocessed_data
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers import Embedding, Input, Dropout
from keras.models import Model

INPUT_DIM = 22000
SENT_LEN = 1000
EMBD_DIM = 128


if __name__ == "__main__":
    inputs_, outputs_ = load_preprocessed_data(sequence_length=SENT_LEN)

    inputs = Input(shape=(SENT_LEN,), dtype="int32")
    embeddings = Embedding(input_dim=INPUT_DIM, output_dim=EMBD_DIM,
                           input_length=SENT_LEN)(inputs)
    lstm1 = Bidirectional(LSTM(units=64))(embeddings)
    dropout = Dropout(0.35)(lstm1)
    outputs = Dense(1, activation="sigmoid")(dropout)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(x=inputs_, y=outputs_, validation_split=0.20, batch_size=64,
              epochs=3)


# ADAM optimizer
# Epoch 3/3
# 20000/20000 [==============================] - 183s - loss: 0.3373 -
# acc: 0.8891 - val_loss: 0.4146 - val_acc: 0.8274

# RMSPROP optimizer
# Epoch 3/3
# 21250/21250 [==============================] - 287s - loss: 0.3246 -
# acc: 0.8913 - val_loss: 0.3591 - val_acc: 0.8579

# ADAM optimizer with dropout 0.35 (too much to generalize?)
# Epoch 3/3
# 20000/20000 [==============================] - 131s - loss: 0.1092 -
# acc: 0.9625 - val_loss: 0.4861 - val_acc: 0.8422

# DROPOUT
# ADAM optimizer with input sequence_length increased to 500 [Acc improved]
# Epoch 2/3
# 20000/20000 [==============================] - 614s - loss: 0.2401 -
# acc: 0.9098 - val_loss: 0.3121 - val_acc: 0.8734
# Epoch 3/
# 20000/20000 [==============================] - 608s - loss: 0.1540 -
# acc: 0.9460 - val_loss: 0.3350 - val_acc: 0.8696

# ADAM without dropout and sequence_length increased to 500 from 100
# Epoch 2/3
# 20000/20000 [==============================] - 414s - loss: 0.2420
#  - acc: 0.9062 - val_loss: 0.3212 - val_acc: 0.8718
# Epoch 3/3
# 20000/20000 [==============================] - 415s - loss: 0.1409
#  - acc: 0.9512 - val_loss: 0.3528 - val_acc: 0.8666

# DROPOUT
# ADAM optimizer with input sequence_length increased to 1000 [Acc improved]
# Epoch 2/3
# 20000/20000 [==============================] - 1244s - loss: 0.2193
# - acc: 0.9190 - val_loss: 0.3382 - val_acc: 0.8524
# Epoch 3/3
# 20000/20000 [==============================] - 1234s - loss: 0.1317
# - acc: 0.9552 - val_loss: 0.3429 - val_acc: 0.8656

# CONCLUSION: Average review length is about 229, so 100 is losing a lot
# of info, likewise 500-1000 doesn't have much info gain. So accuracy is same.
