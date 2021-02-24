import pandas as pd
import nltk # the Natural Language Toolkit, used for preprocessing
from nltk.stem.porter import *
import string # used for preprocessing
import re # used for preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("../data/2500_dataset.csv",low_memory=False)

### Data improvement ###
# Remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
# Rename columns
data.columns = ['user_id', 'artistname', 'trackname', 'playlistname']
# Remove ';;;;' from the column 'playlistname'
data['playlistname'] = data['playlistname'].str.replace(r';;;;', '')

# Conversion to string
data['artistname'] = data['artistname'].astype(str)
data['trackname'] = data['trackname'].astype(str)
data['playlistname'] = data['playlistname'].astype(str)
# Now we can work with our improved dataset

### Explore the 25000_dataset ###
users = data['user_id'].unique() # 30 unique user_ids
artistname = data['artistname'].unique() # 5271 unique artistnames
trackname = data['trackname'].unique() # 18275 unique tracknames
playlistname = data['playlistname'].unique() # 466 unique playlistnames
# print('column_name =',len(users))
# print('column_name =',len(artistname))
# print('column_name =',len(trackname))
# print('column_name =',len(playlistname))

# We check if any value is NaN in our dataset
# num_of_nans = data.isnull().sum()
# print(num_of_nans)
# user_id         0
# artistname      0
# trackname       0
# playlistname    0

# There aren't any NaN values

### Text preprocessing ###
# Function that can be used to preprocess a text
def preprocessing(text):
    # 1. Remove non-letters
    letters_only = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

    # 2. Convert to lower case, split into individual words
    words = letters_only.lower()

    # 3. Remove numbers
    result = re.sub(r'\d+', '', words)

    # 4. Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    no_punctuation = result.translate(translator)

    # 5. Tokenize -> splitting strings into tokens (nominally words). It splits tokens based on white space
    tokenized = word_tokenize(no_punctuation)

    # 6. Remove Stopwords. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    # 7. Remove stop words
    meaningful_words = [w for w in tokenized if not w in stops]  # returns a list

    # 8. Stem words. Need to define porter stemmer above
    stemmer = PorterStemmer()
    singles = [stemmer.stem(word) for word in meaningful_words]

    # 9. Join the words back into one string separated by space, and return the result.
    return (" ".join(singles))


## Preprocessed text to column function ##
def preprocessed_column(column_name):
    preprocessed_text = []
    for text_data in data[column_name]:  # each row in the column will be preprocessed
        pp_text_data = preprocessing(text_data)
        preprocessed_text.append(pp_text_data)
    data[f'pp_text_{column_name}'] = preprocessed_text

preprocessed_column('artistname')
preprocessed_column('trackname')
preprocessed_column('playlistname')

# the columns that we have now are:
# ['user_id', 'artistname', 'trackname', 'playlistname',
# 'pp_text_artistname', 'pp_text_trackname', 'pp_text_playlistname']
# columns with a name that starts with 'pp_text_' have preprocessed text from the original columns#

data['comb'] = data['pp_text_playlistname'] + ' ' + data['pp_text_artistname']

# TfidfVectorizer will convert our data['comb'] (a text column ) into numerical
tfv = TfidfVectorizer()
# TF-IDF matrix with fit_transform method
tfv_matrix = tfv.fit_transform(data['comb'])
# Now that we have a matrix of our words, we can begin calculating similarity scores
# Compute the cosine similarity , a measure of the similarity between 2 vectors
# Use the linear kernel because it is faster
sig = linear_kernel(tfv_matrix, tfv_matrix)
# Now we have the similarity matrix

# Function that gets as inputs the song title and the number of songs requested to appear
def recommend(track_title, how_many):

    # Check if the song title that has been requested is in our dataset
    if track_title not in data['trackname'].unique():
        print('The song title is not in our dataset')
    else:
        i = data.loc[data['trackname'] == track_title].index[0]
        # Get the row related to the trackname from the similarity score matrix
        # Sort the similarity score list on the basis of similarity score
        lst = list(enumerate(sig[i]))
        # The enumerate() function assigns an index to each item in an iterable object that can be used to reference the item later
        # Now we have the indexes of most similar tracknames
        # We need to iterate through the list and store tracknames on the indexes in a new list
        # lst has the index of the trackname and the cosine similarity that the track name has with the track name given
        lst = sorted(lst,key=lambda x: x[1], reverse=True)
        # The list is sorted in the descending order of similarity score
        # We keep only the top (how_many) values of list
        # We remove the track name that the user asked from the lst
        lst = [(val, key) for (val, key) in lst if val != i]
        lst = lst[0:how_many]
        # track_indices is the list with the track names that are going to be returned
        track_indices = []
        for i in range(len(lst)):
            # a gets the index of the tracknames from lst
            index_fom_lst=lst[i][0]
            # We save in track_indices list the track name which index is kept in index_fom_lst
            track_indices.append(data['trackname'][index_fom_lst])
        for i in range(len(track_indices)):
            print(lst[i][0], track_indices[i])


# Ask user the song title and how many results to return
# yes_no = 1
# while yes_no == 1:
#     try:
#         input_song = input("Please enter a song title: ")
#         posa = int(input("Please enter how many songs to return: "))
#         recommend(input_song, posa)
#         print('\n')
#         yes_no = int(input("If you want to continue type 1 else type 0: "))
#     except ValueError:
#        print("\nPlease only use integers")
#        yes_no = int(input("If you want to continue type 1 else type 0: "))
#     else:
#         if yes_no != 0 and yes_no != 1:
#             break



prompt = "if you want the system to recommend a song type 1 else type 0: "
while True:
    try:
        imp = int(input(prompt))
        if imp == 1:
            input_song = input("Please enter a song title: ")
            posa = int(input("Please enter how many songs to return: "))

            try:
                if isinstance(posa, int) == True:
                    recommend(input_song, posa)
                    raise ValueError
                break
            except ValueError:
                prompt = "if you want the system to recommend a song type 1 else type 0: "
            print('\n')
            raise ValueError
        break
    except ValueError:
        prompt = "if you want the system to recommend a song type 1 else type 0: "