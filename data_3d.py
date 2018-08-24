import os
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Dataset params
NUM_SENT = 5
SENT_LEN = 100

# Model params
EMB_DIM = 100
MAX_FET = 22000


def splitlines_smart(string_in):
    inital_list = string_in.splitlines()
    dot_list_ = [each_string.split(".") for each_string in inital_list
                 if len(each_string) > 2]
    dot_list = []
    for each_list in dot_list_:
        dot_list += each_list

    que_list_ = [each_string.split("?") for each_string in dot_list
                 if len(each_string) > 2]
    que_list = []
    for each_list in que_list_:
        que_list += each_list

    exc_list_ = [each_string.split("!") for each_string in que_list
                 if len(each_string) > 2]
    exc_list = []
    for each_list in exc_list_:
        exc_list += each_list

    return exc_list


def load_and_preprocess_data(sent_len=SENT_LEN, num_sent=NUM_SENT):

    local_file_path = "/Users/dsp/Documents/AllProjects/Personal/LearningKeras/old_data/testData.p"
    if os.path.isfile(local_file_path):
        with open(local_file_path, "rb") as data_file:
            reviews, lables = pickle.load(data_file)
    else:
        with open("testData.p", "rb") as data_file:
            reviews, lables = pickle.load(data_file)

    reviews_mask_shape = (len(reviews), num_sent, sent_len)
    reviews_mask = np.zeros(reviews_mask_shape, dtype=np.int32)

    tokenizer = Tokenizer(num_words=MAX_FET)
    tokenizer.fit_on_texts(reviews)

    reviews_lines = [[line for line in splitlines_smart(review) if len(line)]
                     for review in reviews]
    reviews_sequences = [tokenizer.texts_to_sequences(review_lines)
                         for review_lines in reviews_lines]
    reviews_sequences = [pad_sequences(review_sequences, maxlen=sent_len)
                         for review_sequences in reviews_sequences]

    for review_id, review in enumerate(reviews_sequences):
        num_sent_, sent_len_ = review.shape
        reviews_mask[review_id, :num_sent_, :sent_len_] = review[:num_sent, :sent_len]

    _file_name = "preprocessedTestData3D" + str(sent_len) \
                                          + "_" + str(num_sent) + ".p"
    with open(_file_name, "wb +") as data_out:
        pickle.dump([reviews_mask, np.array(lables)], data_out)

    return [reviews_mask, np.array(lables)]


def load_preprocessed_data(sent_len=SENT_LEN, num_sent=NUM_SENT):
    _file_name = "preprocessedTestData3D" + str(sent_len) \
                                          + "_" + str(NUM_SENT) + ".p"

    if os.path.isfile(_file_name) is True:
        print "Loading data..."
        with open(_file_name, "rb") as input_file:
            return pickle.load(input_file)
    else:
        print "Preparing and loading data, this may take a while..."
        return load_and_preprocess_data(sent_len=sent_len, num_sent=num_sent)


if __name__ == "__main__":
    load_preprocessed_data(sent_len=50, num_sent=20)
