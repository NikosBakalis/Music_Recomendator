import functions
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial

if not os.path.isfile("../data/correct_dataset.csv"):
    functions.fix_dataset("../data/spotify_dataset.csv")

X, y = functions.load_dataset("../data/correct_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_enc, X_test_enc = functions.prepare_inputs(X_train, X_test)

y_train_enc, y_test_enc = functions.prepare_inputs(y_train, y_test)

y_train_enc = y_train_enc.ravel()

y_test_enc = y_test_enc.ravel()

knn = KNeighborsClassifier()

knn.fit(X_train_enc, y_train_enc)

prediction = knn.predict(X_test_enc)

result = 1 - spatial.distance.cosine(prediction, y_test_enc)

print(result * 100, "%")
