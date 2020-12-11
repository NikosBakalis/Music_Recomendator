import functions
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# pandas.set_option('display.max_columns', None)

# Read the .csv file and turn it to a dataframe.
# df = pandas.read_csv("../data/spotify_dataset.csv", low_memory=False)
#
# # Take the columns of the dataframe we want as lists.
# userId = df[df.columns[0]].tolist()
# artistName = df[df.columns[1]].tolist()
# trackName = df[df.columns[2]].tolist()
# playlistName = df[df.columns[3]].tolist()
#
# # And pass this columns to a new dataframe.
# df = pandas.DataFrame(list(zip(userId, artistName, trackName, playlistName)), columns=['User_Id', 'Artist_Name',
#                                                                                        'Track_Name', 'Playlist_Name'])
#
# # print(df)
#
# # TILL HERE ALL GOOD!!!
#
# # Gets all unique user IDs.
# uniqueUserId = list(set(userId))
#
# # Creates the song list.
# songList = list(zip(artistName, trackName))
#
# # Gets all unique songs.
# uniqueSongs = list(set(songList))


# String data called categorical data.
# What we have to do is to encode this categorical data.
# The link below will help with the encoding.
# https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
# After we encode the categorical data we can use them to train and test our dataframe.

X, y = functions.load_dataset("../data/spotify_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# prepare input data
X_train_enc, X_test_enc = functions.prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = functions.prepare_targets(y_train, y_test)

# Define the model.
model = Sequential()
model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_enc, y_train_enc, epochs=100, batch_size=16, verbose=2)
# evaluate the keras model
_, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))
