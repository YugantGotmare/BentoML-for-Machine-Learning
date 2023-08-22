from sklearn import datasets
from sklearn import svm

import bentoml as bl

df = datasets.load_iris()
X, y = df.data, df.target

model = svm.SVC(gamma='scale')
model.fit(X,y)

#Save the model to the BentoML local model store
saved_model = bl.sklearn.save_model("iris_clf", model)
print(f"Model Saved: {saved_model}")

## iris_clf:lablorj4k6mtnwga