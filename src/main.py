import pandas
from sklearn.model_selection import train_test_split

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

print(df)

# TILL HERE ALL GOOD!!!

# Get all unique values of userId
uniqueUserId = list(set(userId))
print(uniqueUserId)

songList = list(zip(artistName, trackName, playlistName))
print(len(songList))
# print(songList)
uniqueSongs = list(set(songList))
print(len(uniqueSongs))
