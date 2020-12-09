import pandas

df = pandas.read_csv("../data/spotify_dataset.csv", low_memory=False)

userId = df[df.columns[0]].tolist()
artistName = df[df.columns[1]].tolist()
trackName = df[df.columns[2]].tolist()
playlistName = df[df.columns[3]].tolist()

df = pandas.DataFrame(list(zip(userId, artistName, trackName, playlistName)), columns=['User_Id', 'Artist_Name', 'Track_Name', 'Playlist_Name'])

print(df)
