import pandas

# pandas.set_option('display.max_columns', None)

# Read the .csv file and turn it to a dataframe.
df = pandas.read_csv("../data/spotify_dataset.csv", low_memory=False)

# Take the columns of the dataframe we want as lists.
userId = df[df.columns[0]].tolist()
artistName = df[df.columns[1]].tolist()
trackName = df[df.columns[2]].tolist()
playlistName = df[df.columns[3]].tolist()

# And pass this columns to a new dataframe.
df = pandas.DataFrame(list(zip(userId, artistName, trackName, playlistName)), columns=['User_Id', 'Artist_Name',
                                                                                       'Track_Name', 'Playlist_Name'])

# print(df)

# TILL HERE ALL GOOD!!!

# Gets all unique user IDs.
uniqueUserId = list(set(userId))

# Creates the song list.
songList = list(zip(artistName, trackName))

# Gets all unique songs.
uniqueSongs = list(set(songList))

# String data called categorical data.
# What we have to do is to encode this categorical data.
# The link below will help with the encoding.
# https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
# After we encode the categorical data we can use them to train and test our dataframe.
