import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../data/spotify_dataset.csv",low_memory=False)

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

def plot(column_name):

    plt.subplots(figsize=(12,10))
    list1 = []
    for index, row in column_name.astype(str).items():
    # for i in data['artistname']:
        list1.append(row)
    ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
    for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):
        ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
    plt.title('Top')
    plt.show()

plot(data['artistname'])
plot(data['trackname'])
plot(data['playlistname'])


