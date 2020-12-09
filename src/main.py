import pandas

df = pandas.read_csv("../data/spotify_dataset.csv", low_memory=False)

userId = df[df.columns[0]]
artistName = df[df.columns[1]]
trackName = df[df.columns[2]]
playlistName = df[df.columns[3]]

print(userId)

print("re")
