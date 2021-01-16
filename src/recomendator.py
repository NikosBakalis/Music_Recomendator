import pandas as pd
from sklearn.model_selection import train_test_split
import plots
import matplotlib.pyplot as plt
import numpy as np

# content based recommender systems python

data = pd.read_csv("../data/small_dataset.csv",low_memory=False)

### data improvement ###

#remove unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#rename columns
data.columns = ['user_id', 'artistname', 'trackname', 'playlistname']
#remove ';;;;' from the column 'playlistname'
data['playlistname'] = data['playlistname'].str.replace(r';;;;', '')
# print(data)
#now we can work with our improved dataset


### explore dataset ###

users = data['user_id'].unique() # 562 unique user_ids
artistname = data['artistname'].unique() # 35023 unique artistnames
trackname = data['trackname'].unique() # 190106 unique tracknames
playlistname = data['playlistname'].unique() # 7117 unique playlistnames
# print('column_name =',len(column_name))

# we check if any value is NaN in our dataset
num_of_nans = data.isnull().sum()
# print(num_of_nans)
# user_id           0
# artistname      828
# trackname       109
# playlistname    696

#different ways to fill NaN values:
# https://medium.com/analytics-vidhya/ways-to-handle-categorical-column-missing-data-its-implementations-15dc4a56893
#https://medium.com/analytics-vidhya/best-way-to-impute-categorical-data-using-groupby-mean-mode-2dc5f5d4e12d


# plots.plot(data['trackname'])

data.dropna(inplace=True)
# print(data.isnull().sum())

# https://medium.com/analytics-vidhya/content-based-recommender-systems-in-python-2b330e01eb80

# # https://heartbeat.fritz.ai/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831
# ### Creating a TF-IDF Vectorizer ###
# # TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
# # IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
#
# tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
# tfidf_matrix = tf.fit_transform(data['playlistname'].values.astype('U'))
#
# # need to convert the dtype object to unicode string -> .astype('U')
# # print(tfidf_matrix)
# # Here, the tfidf_matrix is the matrix containing each word and
# # its TF-IDF score with regard to each document, or item in this case.
# # Also, stop words are simply words that add no significant value to our system,
# # like ‘an’, ‘is’, ‘the’, and hence are ignored by the system.
#
# # we need to calculate the relevance or similarity of one document to another
# # Vector Space Model
# # Calculating Cosine Similarity
# cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
# results = {}
# for idx, row in data.iterrows():
#    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
#    similar_items = [(cosine_similarities[idx][i], data['user_id'][i]) for i in similar_indices]
#    results[row['user_id']] = similar_items[1:]
#
# def item(id):
#   return data.loc[data['user_id'] == id]['playlistname'].tolist()[0].split(' - ')[0]
#   # Just reads the results out of the dictionary.def
# def recommend (item_id, num):
#     print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
#     print("-------")
#     recs = results[item_id][:num]
#     for rec in recs:
#        print("Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")
#
# recommend(item_id=11, num=5)


# # # https://www.kdnuggets.com/2019/11/content-based-recommender-using-natural-language-processing-nlp.html
# from rake_nltk import Rake
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
#
# data['Key_words'] = ''
# r = Rake()
# for index, row in data.iterrows():
#     r.extract_keywords_from_text(row['artistname'])
#     key_words_dict_scores = r.get_word_degrees()
#     row['Key_words'] = list(key_words_dict_scores.keys())
#
# data['trackname'] = data['trackname'].map(lambda x: x.split(','))
# data['artistname'] = data['artistname'].map(lambda x: x.split(',')[:3])
# data['playlistname'] = data['playlistname'].map(lambda x: x.split(','))
# for index, row in data.iterrows():
#     row['trackname'] = [x.lower().replace(' ','') for x in row['trackname']]
#     row['artistname'] = [x.lower().replace(' ','') for x in row['artistname']]
#     row['playlistname'] = [x.lower().replace(' ','') for x in row['playlistname']]
#
# data['Bag_of_words'] = ''
# columns = ['artistname']
# for index, row in data.iterrows():
#     words = ''
#     for col in columns:
#         words += ' '.join(row[col]) + ' '
#     row['Bag_of_words'] = words
#
# df = data[['artistname', 'Bag_of_words']]
# # print(data[['trackname', 'Bag_of_words']]
#
# count = CountVectorizer()
# count_matrix = count.fit_transform(df['Bag_of_words'])
# cosine_sim = cosine_similarity(count_matrix, count_matrix)
# # print(cosine_sim)
#
# indices = df['artistname']
#
# def recommend(title, cosine_sim=cosine_sim):
#     recommended_movies = []
#     idx = indices[indices == title].index[0]
#     score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
#     top_10_indices = list(score_series.iloc[1:11].index)
#
#     for i in top_10_indices:
#         recommended_movies.append(list(df['artistname'])[i])
#
#     return recommended_movies
#
# recommend('elviscostello')

# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
#
# matrix = tf.fit_transform(data['trackname'])
# from sklearn.metrics.pairwise import linear_kernel
#
# cosine_similarities = linear_kernel(matrix,matrix)
#
# movie_title = data['artistname']
# indices = pd.Series(data.index, index=data['artistname'])
#
# def movie_recommend(original_title):
#     idx = indices[original_title]
#     sim_scores = list(enumerate(cosine_similarities[idx]))
#     sim_scores = sorted(sim_scores)
#     sim_scores = sim_scores[1:31]
#     movie_indices = [i[0] for i in sim_scores]
#     print(idx)
#     return movie_title.iloc[movie_indices]
#
# movie_recommend('Elvis Costello')