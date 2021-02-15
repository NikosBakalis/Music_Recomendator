import pandas
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# Fixes the dataset.
def fix_dataset(filename):
    df = pandas.read_csv(filename)
    df = df.iloc[:, 0:4]
    df.rename(columns={df.columns[0]: "User_Id",
                       df.columns[1]: "Artist_Name",
                       df.columns[2]: "Track_Name",
                       df.columns[3]: "Playlist_Name"},
              inplace=True)
    df.dropna(inplace=True)
    df["Playlist_Name"] = df["Playlist_Name"].map(lambda x: x.rstrip(';'))
    df.to_csv("../data/correct_dataset.csv", index=False)


# Load the dataset.
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = pandas.read_csv(filename, low_memory=False)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, [1, 2, 3]]
    y = dataset[:, [0]]
    # format all fields as string
    X = X.astype(str)
    # reshape target to be a 2d array
    # y = y.reshape((len(y), 1))
    return X, y


# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    # print(type(X_train))
    X_train_enc = oe.transform(X_train)
    oe.fit(X_test)
    # print(type(X_test))
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    le.fit(y_test)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
