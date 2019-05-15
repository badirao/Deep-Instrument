import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv("proj.csv")
X = data[['TABLA', 'SHANKA', 'SITAR', 'VEENA','BHANSURI','GHATAM']]
Y = data[['LABEL']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)


# Now apply the transformations to the data:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=800)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
target_names = ['hindustani','carnatic']
print accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=target_names))


