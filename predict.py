import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import spacy
import pickle
import string
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import LdaMulticore, TfidfModel, CoherenceModel
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import FAST_VERSION
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases

from keras import backend as K
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Input, Embedding
from keras.layers import Bidirectional, LSTM
from keras.layers import Dropout, Dense, Activation
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

###### RECONSTRUCTING MODELS ######


def f1_multiclass(y_true, y_pred):
    """
    Compute the F-score for multiclass labels
    """

    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    y_true_arg = y_true.argmax(axis=2)
    y_pred_arg = y_pred.argmax(axis=2)

    f1s = np.zeros((y_true.shape[-1]))

    for i in range(y_true.shape[2]):
        tp = np.sum(np.logical_and((y_true_arg == i), (y_pred_arg == i)))
        fp = np.sum(np.logical_and((y_true_arg != i), (y_pred_arg == i)))
        fn = np.sum(np.logical_and((y_true_arg == i), (y_pred_arg != i)))

        precision = tp / (tp + fp + np.finfo(float).eps)
        recall = tp / (tp + fn + np.finfo(float).eps)
        f1s[i] = 2 * ((precision * recall) / (precision + recall + np.finfo(float).eps))

    return round(f1s.mean(), 2)


# NER dictionaries
word2int, prefix2int, suffix2int, int2tag = pickle.load(
    open("models/corpus.pickle", "rb")
)

# reconstructing Bi-LSTM model
model = keras.models.load_model(
    "models/bilstm_best", custom_objects={"f1_multiclass": f1_multiclass}
)

# reconstructing the SVM classifier
tfidf_vectorizer, svm = pickle.load(open("models/SVM_classification.pickle", "rb"))

############################################


###### SETTING UP PREDICTION PIPELINE ######


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=argparse.FileType("r"),
        help="Input your recipe .txt file here",
    )

    opt = parser.parse_args()
    return opt


def extract_ingredient(recipe, model):

    recipe = [
        token.text for token in nlp(recipe) if token.text not in string.punctuation
    ]

    output = []
    for batch in [
        recipe[i : min(len(recipe), i + 21)] for i in range(0, len(recipe), 21)
    ]:

        # convert to integers
        recipe_int = [[word2int.get(word, 1) for word in batch]]
        # replace prefixes
        prefix_ints = [[prefix2int.get(word[:3], 1) for word in batch]]
        # replace suffixes
        suffix_ints = [[suffix2int.get(word[-3:], 1) for word in batch]]

        # add padding
        recipe_int = pad_sequences(recipe_int, maxlen=22, padding="post")
        prefix_int = pad_sequences(prefix_ints, maxlen=22, padding="post")
        suffix_int = pad_sequences(suffix_ints, maxlen=22, padding="post")

        predictions = model.predict([recipe_int, prefix_int, suffix_int])

        for prediction in predictions:
            token_sequence = [
                int2tag[np.argmax(prediction[i])] for i in range(len(batch))
            ]

        for item, tag in zip(batch, token_sequence):
            if tag not in set({"-PAD-", "O"}):
                output.append(item)

    return " ".join(output).strip()


def process_string(recipe):
    prep_r = []
    for ing in recipe.split():
        ing = [WordNetLemmatizer().lemmatize(word) for word in str(ing).split()]
        ing = " ".join(ing)
        ing = re.sub(
            r"\(.*oz.\)|crushed|crumbles|ground|minced|powder|chopped|sliced|and|or",
            "",
            ing,
        )
        ing = re.sub("[^a-zA-Z]", " ", ing)
        ing = ing.lower().strip()
        prep_r.append(ing)

    return [" ".join(prep_r)]


def predict(recipe):

    int_output = extract_ingredient(recipe, model)
    int_output = process_string(int_output)
    # print("Ingredients detected: ", int_output[0].split())
    int_output = tfidf_vectorizer.transform(int_output)
    output = svm.predict(int_output)

    return output[0]


if __name__ == "__main__":
    opt = parse_opt()
    recipe = " ".join(opt.recipe.readlines())
    print("This recipe might be {}".format(predict(recipe)))
