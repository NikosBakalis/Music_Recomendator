# region Imports

import functions
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial

# endregion

# region Checks if the correct file exists.

# If the path does NOT exists...
if not os.path.isfile("../data/correct_dataset.csv"):
    # Calls function to fix the dataset.
    functions.fix_dataset("../data/spotify_dataset.csv")

# endregion

# region Loads and encodes the dataset.

X, y = functions.load_dataset("../data/correct_dataset.csv")

X_enc, y_enc = functions.prepare_inputs(X, y)

# endregion

# region Splits the dataset to train and test.

X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(X_enc, y_enc, train_size=0.75,  shuffle=True)

X_combine = numpy.stack((X_enc, X), axis=1)

y_combine = numpy.stack((y_enc, y), axis=1)

y_train_enc = y_train_enc.ravel()

y_test_enc = y_test_enc.ravel()

# region Creation of dictionaries.

X_combine = X_combine.ravel()

X_combine_dict = {(X_combine[i], X_combine[i + 1], X_combine[i + 2]): [X_combine[i + 3], X_combine[i + 4], X_combine[i + 5]] for i in range(0, len(X_combine), 6)}

y_combine = y_combine.ravel()

y_combine_dict = dict(zip(y_combine[::2], y_combine[1::2]))

# endregion

# endregion

# region K-Nearest-Neighbors.

knn = KNeighborsClassifier()

knn.fit(X_train_enc, y_train_enc)

prediction = knn.predict(X_test_enc)

result = 1 - spatial.distance.cosine(prediction, y_test_enc)

print(y_test_enc[0], "\n")

print((str(X_test_enc[0][0]), str(X_test_enc[0][1]), str(X_test_enc[0][2])))

print(X_combine_dict.get((str(X_test_enc[0][0]), str(X_test_enc[0][1]), str(X_test_enc[0][2]))), "\n")

print(prediction[0])

print(y_combine_dict.get(prediction[0]), "\n")

print(round(result * 100, 2), "%")

# endregion
