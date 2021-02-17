import functions
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial

if not os.path.isfile("../data/correct_dataset.csv"):
    functions.fix_dataset("../data/spotify_dataset.csv")

X, y = functions.load_dataset("../data/correct_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,  shuffle=True)

X_train_enc, X_test_enc = functions.prepare_inputs(X_train, X_test)

X_train_combine = numpy.stack((X_train_enc, X_train), axis=1)

X_test_combine = numpy.stack((X_test_enc, X_test), axis=1)

y_train_enc, y_test_enc = functions.prepare_inputs(y_train, y_test)

y_train_combine = numpy.stack((y_train_enc, y_train), axis=1)

y_test_combine = numpy.stack((y_test_enc, y_test), axis=1)

y_train_enc = y_train_enc.ravel()

y_test_enc = y_test_enc.ravel()

knn = KNeighborsClassifier()

knn.fit(X_train_enc, y_train_enc)

prediction = knn.predict(X_test_enc)

test_prediction = knn.predict(X_test_enc)

# print(y_test_enc[0:50])
#
# print(test_prediction[0:50])

result = 1 - spatial.distance.cosine(prediction, y_test_enc)

print(round(result * 100, 2), "%")
