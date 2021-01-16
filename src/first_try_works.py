import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


# we check if any value is NaN in our dataset
data.dropna(inplace=True)


# https://www.academyofdatascience.com/Blog_page/blog_3.html
# douleuei alla den ksero an einai auto poy theloume

data['trackname'] = data['trackname'].str.lower()
# print(data['trackname'].head())
data['comb'] = data['artistname'] + ' ' + data['playlistname']
# print(data['comb'].head(10))

# count matrix of features and then similarity Score matrix.
# Similarity score matrix contains the cosine similarity of all the
# movies in the dataset. We can get the similarity score of two
# trackanames by there index in dataset.

cv=CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])
sim=cosine_similarity(count_matrix)
# print(sim)

def recommend(m,how_many):
# string trackname kai posa idia thelo na bgalei
    m=m.lower()
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