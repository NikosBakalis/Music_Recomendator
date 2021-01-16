import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("../data/spotify_dataset.csv",low_memory=False)

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

# plt.subplots(figsize=(12,10))
# list1 = []
# for index, row in data['playlistname'].astype(str).items():
# # for i in data['artistname']:
#     list1.append(row)
# ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
# for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):
#     ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
# plt.title('Top Playlists')
# plt.show()
#
# plt.subplots(figsize=(12,10))
# list1 = []
# for index, row in data['trackname'].astype(str).items():
# # for i in data['artistname']:
#     list1.append(row)
# ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
# for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):
#     ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
# plt.title('Top Tracks')
# plt.show()