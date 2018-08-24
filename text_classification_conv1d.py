from keras.layers import Conv1D, MaxPooling1D, Dense
from keras.layers import Embedding, Input, Flatten, Dropout
from keras.models import Model
from data import load_preprocessed_data

SEQ_LEN = 1000
EMB_DIM = 100
MAX_FET = 22000


if __name__ == "__main__":

    inputs, outputs = load_preprocessed_data()

    input = Input(shape=(SEQ_LEN,), dtype="int32")
    embedding = Embedding(input_dim=MAX_FET, output_dim=EMB_DIM,
                          input_length=SEQ_LEN)(input)
    conv1 = Conv1D(filters=128, kernel_size=5, activation="relu")(embedding)
    pool1 = MaxPooling1D(pool_size=5)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=5, activation="relu")(pool1)
    pool2 = MaxPooling1D(pool_size=5)(conv2)
    conv3 = Conv1D(filters=128, kernel_size=5, activation="relu")(pool2)
    pool3 = MaxPooling1D(pool_size=35)(conv3)

    flattened = Flatten()(pool3)
    dropout = Dropout(rate=0.5)(flattened)
    fully_connected = Dense(32, activation="relu")(dropout)
    output = Dense(1, activation="sigmoid")(fully_connected)

    model = Model(inputs=input, outputs=output)
    model.summary()  # Run Hydrogen Above all here to see model output

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.fit(x=inputs, y=outputs, batch_size=256,
              epochs=3, validation_split=0.15)


# Without dropout
# 21250/21250 [==============================] - 181s - loss: 0.2051 -
# acc: 0.9196 - val_loss: 0.2955 - val_acc: 0.8819

# With dropout
# 21250/21250 [==============================] - 183s - loss: 0.2267
# acc: 0.9129 - val_loss: 0.3761 - val_acc: 0.8539
