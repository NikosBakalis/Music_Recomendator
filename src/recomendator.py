import pandas as pd
import plots

# content based recommender systems python

data = pd.read_csv("../data/spotify_dataset.csv",low_memory=False)

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

plots.plot(data['artistname'])
plots.plot(data['trackname'])
plots.plot(data['playlistname'])


# We can create a song recommender by splitting our dataset
# into training and testing data.
# We use the train_test_split function of scikit-learn library.

# train_data, test_data = train_test_split(data, test_size = 0.20, random_state=0)