import pandas as pd

# preprocessed text and recommendation

# TO_DO: content based recommender systems python evaluation metrics


data = pd.read_csv("../data/small_dataset.csv", low_memory=False)

### data improvement ###

# remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
# rename columns
data.columns = ['user_id', 'artistname', 'trackname', 'playlistname']
# remove ';;;;' from the column 'playlistname'
data['playlistname'] = data['playlistname'].str.replace(r';;;;', '')
# print(data)
# now we can work with our improved dataset


# processing of overviews
# user_id           0
# artistname      828
# trackname       109
# playlistname    696

# https://morioh.com/p/2fea5a49b62d

# import basic libraries
import pandas as pd
from nltk.stem.porter import *

stemmer = PorterStemmer()
# load the word2vec algorithm from the gensim library
# from gensim.models import word2vec

# def review_to_words(raw_review):
#     # 1. Remove non-letters
#     letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
#     # 2. Convert to lower case, split into individual words
#     words = letters_only.lower().split()
#
#     # 3. Remove Stopwords. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
#     stops = set(stopwords.words("english"))
#
#     # 4. Remove stop words
#     meaningful_words = [w for w in words if not w in stops]  # returns a list
#
#     # 5. Stem words. Need to define porter stemmer above
#     singles = [stemmer.stem(word) for word in meaningful_words]
#
#     # 6. Join the words back into one string separated by space, and return the result.
#     return (" ".join(singles))
#
# # print(review_to_words)
# # print(data['trackname'].head(10))
# # data['trackname'] = data.trackname.apply(review_to_words)
# # print(data['trackname'].head(10))

# kano apo edo :
# https://towardsdatascience.com/preprocessing-text-data-in-python-an-introduction-via-kaggle-7d28ad9c9eb
# thelo na ftiakso ta strings tis katse stils gia na mporei na ginei kalitero to tfidf
# diladi o ipologismos
# meta to recomend to afino idio i perno apo to first_try

# https://towardsdatascience.com/preprocessing-text-data-in-python-an-introduction-via-kaggle-7d28ad9c9eb
#

import string  # used for preprocessing
import re  # used for preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # used for preprocessing
from nltk.stem import WordNetLemmatizer  # used for preprocessing


def remove_urls(text):
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+) | ([^0-9A-Za-z \t]) | (\w+:\/\/\S+)", " ", text).split())
    return new_text  # make all text lowercase


def text_lowercase(text):
    return text.lower()  # remove numbers


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result  # remove punctuation


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)  # tokenize


def tokenize(text):
    text = word_tokenize(text)
    return text  # remove stopwords


stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text  # lemmatize


lemmatizer = WordNetLemmatizer()


def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text


def preprocessing(text):
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text


# print(data['trackname'].head(10))
# data['trackname'] = data.trackname.apply(preprocessing)
# print(data['trackname'].head(10))

def pp_text_column(column_name):
    pp_text_train = []  # our preprocessed text column
    for text_data in data[column_name]:
        pp_text_data = preprocessing(text_data)
        pp_text_train.append(pp_text_data)
    data[f'pp_text_{column_name}'] = pp_text_train


pp_text_column('trackname')
pp_text_column('playlistname')
pp_text_column('artistname')
# print(data[['artistname','pp_text_artistname']].head(10))
# print(data.columns)
from sklearn.feature_extraction.text import TfidfVectorizer

# Using Abhishek Thakur's arguments for TF-IDF
# tfv = TfidfVectorizer(min_df=3,  max_features=None,
#             strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#             ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
#             stop_words = 'english')

tfv = TfidfVectorizer()

# Filling NaNs with empty string
# data['trackname'] = data['trackname'].fillna('')
data['comb'] = data['pp_text_playlistname'] + ' ' + data['pp_text_artistname']

tfv_matrix = tfv.fit_transform(data['comb'])

print(tfv_matrix.shape)
# print(tfv_matrix)

from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(data.index, index=data['trackname']).drop_duplicates()


# Credit to Ibtesam Ahmed for the skeleton code
def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return data['trackname'].iloc[movie_indices]


print(give_rec('7 Years Too Late'))
