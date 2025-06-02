import numpy as np
import pandas as pd
from DecisionTree import DecisionTree

def initialise():
    # Load training features
    X_train = pd.read_csv("traindata.txt", header=None, sep=',')

    #deal with non-numericals in data set
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    # Load training labels
    y_train = pd.read_csv("trainlabels.txt", header=None, names=['label'])

    # print("NaNs in features:", X_train.isna().sum().sum())
    # print("Infs in features:", np.isinf(X_train.values).sum())

    # Data proccesssing
    # X_train = X_train.replace([np.inf, -np.inf], np.nan)
    # X_train = X_train.fillna(X_train.mean())

    # Display shapes to verify
    # print("Training features shape:", X_train.shape)
    # print("Training labels shape:", y_train.shape)
    return X_train,y_train

train_features,train_labels = initialise()

tree = DecisionTree()
tree.fit(train_features,train_labels)
print("DONE FITTING!")
predictions = tree.predict(train_features.values)

def accuracy(y_test,y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

acc = accuracy(train_labels.values.ravel(),predictions)
print("Training accuracy: ",acc)