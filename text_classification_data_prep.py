# Parsing dataset and preparing data.

import re
import pickle
import pandas as pd
from bs4 import BeautifulSoup


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


texts = []
labels = []

with open("labeledTrainData.tsv", "rb") as data_file:
    data_train = pd.read_csv(data_file, delimiter='\t')
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.sentiment[idx])

with open("testData.p", "wb+") as out_file:
    pickle.dump([texts, labels], out_file)
