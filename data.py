import os
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

SEQ_LEN = 100
EMB_DIM = 100
MAX_FET = 22000


def load_and_preprocess_data(sequence_length=SEQ_LEN):
    with open("/Users/dsp/Documents/AllProjects/Personal/LearningKeras/old_data/testData.p", "rb") as data_file:
        reviews, lables = pickle.load(data_file)

    tokenizer = Tokenizer(num_words=MAX_FET)
    tokenizer.fit_on_texts(reviews)

    reviews_sequences = tokenizer.texts_to_sequences(reviews)
    reviews_sequences = pad_sequences(reviews_sequences, maxlen=sequence_length)

    _file_name = "preprocessedTestData" + str(sequence_length) + ".p"
    with open(_file_name, "wb +") as data_out:
        pickle.dump([reviews_sequences, np.array(lables)], data_out)

    return [reviews_sequences, np.array(lables)]


def load_preprocessed_data(sequence_length=SEQ_LEN):
    _file_name = "preprocessedTestData" + str(sequence_length) + ".p"

    if os.path.isfile(_file_name) is True:
        print "Loading data..."
        with open(_file_name, "rb") as input_file:
            return pickle.load(input_file)
    else:
        print "Preparing and loading data, this may take a while..."
        return load_and_preprocess_data(sequence_length=sequence_length)


if __name__ == "__main__":
    load_preprocessed_data(sequence_length=500)
