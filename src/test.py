from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
OrdinalEncoder()
enc.categories_
print(enc.transform([['Female', 3], ['Male', 1], ['Female', 3]]))

print(enc.inverse_transform([[1, 0], [0, 1]]))


