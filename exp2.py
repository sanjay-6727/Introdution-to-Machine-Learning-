from eda_final import eda_analysis
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("spambase_csv.csv")

df=eda_analysis(df)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

x=df.drop("class",axis=1)
y=df["class"]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("\nAccuracy_Score of GNB",accuracy_score(y_test,y_pred))

model=BernoulliNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("\nAccuracy_Score of BNB",accuracy_score(y_test,y_pred))


minmax = MinMaxScaler()

X_train_m = minmax.fit_transform(X_train)
X_test_m = minmax.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_m, y_train)


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=KNeighborsClassifier(n_neighbors=5)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("\nAccuracy_Score of KNN",accuracy_score(y_test,y_pred))


pg={"n_neighbors":[3,5,7,9,11], "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"]}

knn=KNeighborsClassifier()
grid=GridSearchCV(estimator=knn,param_grid=pg,cv=5,scoring="accuracy",n_jobs=-1)#cv is the number of folds for cross validation and n_jobs s the number of fcores to be used and -1 indicates all cores present to be used 

import time
start=time.perf_counter()
grid.fit(X_train,y_train)
end=time.perf_counter()
training_time = end - start

print("Best Parmeters",grid.best_params_)

print("Best CV Score",grid.best_score_)
s=time.perf_counter()
y_pred=grid.predict(X_test)
e = time.perf_counter()
prediction_time =  e- s
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.6f} seconds")
print("Best K:", grid.best_params_["n_neighbors"])
print("\nAccuracy_Score of KNN using GridSearchCV",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#randomizedcv
from sklearn.model_selection import RandomizedSearchCV
pr={"n_neighbors":range(1,31), "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],"p":[1,2]}

rs=RandomizedSearchCV(estimator=knn,param_distributions=pr,n_iter=20,cv=5,scoring="accuracy",n_jobs=-1,random_state=42)#cv is the number of folds for cross validation and n_jobs s the number of fcores to be used and -1 indicates all cores present to be used 

start=time.perf_counter()
rs.fit(X_train,y_train)
end=time.perf_counter()
training_time = end - start

print("Best Parmeters",rs.best_params_)

print("Best CV Score",rs.best_score_)
s=time.perf_counter()
y_pred=rs.predict(X_test)
e = time.perf_counter()
prediction_time =  e- s
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.6f} seconds")
print("Best K:", rs.best_params_["n_neighbors"])
print("\nAccuracy_Score of KNN using RandomizedCV",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


#kdtree


model=KNeighborsClassifier(n_neighbors=5,algorithm="kd_tree")


start = time.perf_counter()
model.fit(X_train, y_train)
train_time = time.perf_counter() - start

start = time.perf_counter()
y_pred = model.predict(X_test)
predict_time = time.perf_counter() - start

print("KDTree")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Training Time:", train_time)
print("Prediction Time:", predict_time)

model=KNeighborsClassifier(n_neighbors=5,algorithm="ball_tree")


start = time.perf_counter()
model.fit(X_train, y_train)
train_time = time.perf_counter() - start

start = time.perf_counter()
y_pred = model.predict(X_test)
predict_time = time.perf_counter() - start

print("BallTree")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Training Time:", train_time)
print("Prediction Time:", predict_time)
