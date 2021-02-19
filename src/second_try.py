# import basic libraries
import pandas as pd
from nltk.stem.porter import *
import string # used for preprocessing
import re # used for preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# import numpy as np # used for managing NaNs
# import nltk # the Natural Language Toolkit, used for preprocessing

# TO_DO: content based recommender systems python evaluation metrics

data = pd.read_csv("../data/25000_dataset.csv",low_memory=False)

### data improvement ###
#remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#rename columns
data.columns = ['user_id', 'artistname', 'trackname', 'playlistname']
#remove ';;;;' from the column 'playlistname'
data['playlistname'] = data['playlistname'].str.replace(r';;;;', '')

#now we can work with our improved dataset
data['artistname'] = data['artistname'].astype(str)
data['trackname'] = data['trackname'].astype(str)
data['playlistname'] = data['playlistname'].astype(str)

### explore dataset ###
users = data['user_id'].unique() # 562 unique user_ids
artistname = data['artistname'].unique() # 35023 unique artistnames
trackname = data['trackname'].unique() # 190106 unique tracknames
playlistname = data['playlistname'].unique() # 7117 unique playlistnames
# print('column_name =',len(column_name))

# we check if any value is NaN in our dataset
num_of_nans = data.isnull().sum()
# print(num_of_nans)
# user_id         0
# artistname      0
# trackname       0
# playlistname    0

#there aren't any NaN values


####################################################################################################
def review_to_words(text):
    # 1. Remove non-letters
    letters_only = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower()

    result = re.sub(r'\d+', '', words)

    translator = str.maketrans('', '', string.punctuation)
    a = result.translate(translator)

    b = word_tokenize(a)

    # 3. Remove Stopwords. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    # 4. Remove stop words
    meaningful_words = [w for w in b if not w in stops]  # returns a list

    lemmatizer = WordNetLemmatizer()
    c = [lemmatizer.lemmatize(token) for token in meaningful_words]

    # 5. Stem words. Need to define porter stemmer above
    stemmer = PorterStemmer()
    singles = [stemmer.stem(word) for word in c]

    # 6. Join the words back into one string separated by space, and return the result.
    return (" ".join(singles))


# print(review_to_words)
# print(data['trackname'].head(10))
# data['trackname'] = data.trackname.apply(review_to_words)
# print(data['trackname'].head(10))

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
####################################################################################################

## preprocessed text to column function ##
def preprocessed_column(column_name):
    preprocessed_text = []
    for text_data in data[column_name]: #each row in the column will be preprocessed
        pp_text_data = review_to_words(text_data)
        preprocessed_text.append(pp_text_data)
    data[f'pp_text_{column_name}'] = preprocessed_text

preprocessed_column('artistname')
preprocessed_column('trackname')
preprocessed_column('playlistname')

# the columns that we have now are:
# ['user_id', 'artistname', 'trackname', 'playlistname',
# 'pp_text_artistname', 'pp_text_trackname', 'pp_text_playlistname']
# columns with a name that starts with 'pp_text_' have preprocessed text from the original columns#

tfv = TfidfVectorizer()
data['comb'] = data['pp_text_playlistname'] + ' ' + data['pp_text_artistname']

tfv_matrix = tfv.fit_transform(data['comb'])
print(tfv_matrix.shape)
# print(tfv_matrix)

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and tracknames
indices = pd.Series(data.index, index=data['trackname']).drop_duplicates()

def give_title(title, sig=sig):
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

print(give_title('7 Years Too Late'))


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data['comb'] = data['pp_text_playlistname'] + ' ' + data['pp_text_artistname']

tfv = TfidfVectorizer()
data['comb'] = data['pp_text_playlistname'] + ' ' + data['pp_text_artistname']
tfv_matrix = tfv.fit_transform(data['comb'])
# print(tfv_matrix.shape)
sim = sigmoid_kernel(tfv_matrix, tfv_matrix)

cv=HashingVectorizer()
count_matrix = cv.fit_transform(data['comb'])
# sim=cosine_similarity(count_matrix)
# print(sim)
# print(type(sim))

def recommend(m,how_many):
# string trackname kai posa idia thelo na bgalei

    if m not in data['trackname'].unique():
        print('not in our dataset')
    else:
        i = data.loc[data['trackname'] == m].index[0] #vrisko pou einai to trackname sto dataset
        # let’s sort the similarity score list on the basis of similarity score
        lst = list(enumerate(sim[i]))
        # now we have the indexes of most similar tracknames
        # we need to iterate through the list and store tracknames on the indexes in a new list
        lst = sorted(lst,key=lambda x:x[1],reverse=True)
        # the list is sorted in the descending order of similarity score
        # keep only the top (how_many) values of list
        #not keeping the first index(0) because its the same trackname
        lst = lst[1:how_many+1]
        l=[]
        for i in range(len(lst)):
            a=lst[i][0]
            l.append(data['trackname'][a])
        for i in range(len(l)):
            print(l[i])


recommend('7 Years Too Late',11)