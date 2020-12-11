from pandas import read_csv
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# Load the dataset.
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, low_memory=False)
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
    print(X_train)
    X_train_enc = oe.transform(X_train)
    print(X_test)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
