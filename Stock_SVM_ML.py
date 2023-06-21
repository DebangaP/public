"""
-> Generates ML models
->. Split into test/train (20/80)
->. Run RF, MLCP etc. (use Scaling if necessary)
->. Check confidence, modify data in 1 and run 1-5, till better confidence
"""

from sklearn.model_selection import train_test_split
import glob
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import datetime
from datetime import timedelta

baseDir = "C:\work\code\market\\"
inputDir = baseDir + "test\\"
modelDir = baseDir + "models\\"
downloadDir = baseDir

#inputDir = "/Users/guest1/Documents/work/Stock-ML/processed/"
#modelDir = "/Users/guest1/Documents/work/Stock-ML/models/"
#downloadDir = "/Users/guest1/Documents/work/Stock-ML/predictions/"
#ma_dir = "/Users/guest1/Downloads/bhavcopyDownloads/"

today = datetime.date.today()
process_date = today

print('1')
print(inputDir)

verdict={
  'A':'Excellent'
  ,'B':'Good'
  ,'C':'Avoid'
}

y=[]
data = []

for file in glob.glob(inputDir + '*.csv'):
    file_name=os.path.basename(file)
    print(file_name)

    df = pd.read_csv(file, index_col=None, header=0)
    data.append(df)

    # fetch already tagged class for training
    v = verdict[file_name.split("_")[1]]
    y.append(v)

X = pd.concat(data, axis=0, ignore_index=True)
X = X.fillna(0)
X.drop_duplicates()

print(X.shape)
print(y)
print(len(y))

#dfgdfgdfg

#X.to_csv(downloadDir + "X.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print("\\n")
print((X_train.shape[0], X_test.shape[0]))
print(f'Features extracted: {X_train.shape}')


#Create a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200)

#Train the model
rf_model.fit(X_train, y_train)

#Predict for the test set
y_pred = rf_model.predict(X_test)

#Calculate the accuracy of the model
rf_accuracy = accuracy_score ( y_true = y_test, y_pred = y_pred)
print("Random Forest Accuracy: {:.2f}%".format( rf_accuracy * 100))


# Multi Layer Perceptron Classifier
# Multi-layer Perceptron Classifier is sensitive to feature scaling

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp_model = MLPClassifier(alpha=0.01, batch_size=64, epsilon=1e-08, hidden_layer_sizes=(300,), 
                    learning_rate='adaptive', max_iter=5000, random_state=0)

#Train the model
mlp_model.fit (X_train, y_train)

#Predict for the test set
y_pred = mlp_model.predict(X_test)
#print(y_pred)

#Calculate the accuracy of the model
mlp_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("MLPC Accuracy: {:.2f}%".format(mlp_accuracy * 100))
#print("\n")

# Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

dt_accuracy = accuracy_score ( y_true = y_test, y_pred = y_pred)
print("Decision Tree prediction accuracy: {:.2f}%".format( dt_accuracy * 100))
print("\n")

_ml_model_file = "Model" + "_rf_" + str(process_date.strftime('%d%m%y')) + ".mdl"

joblib.dump(rf_model, modelDir + _ml_model_file)
