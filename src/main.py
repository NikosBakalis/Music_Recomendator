import functions
import os
from sklearn.model_selection import train_test_split

if not os.path.isfile("../data/correct_dataset.csv"):
    functions.clear_dataset("../data/small_dataset.csv")

X, y = functions.load_dataset("../data/correct_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X_train)
