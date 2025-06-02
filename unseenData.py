import numpy as np
import pandas as pd
from DecisionTree import DecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def initialise():
    data = load_iris()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='label')

    # Fill missing values just in case
    X = X.fillna(X.mean())
    return X, y

# Load data
features, labels = initialise()

# Split into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Initialize and train the DecisionTree on training data 
tree = DecisionTree()
tree.fit(X_train, y_train)
print("DONE FITTING!")

# Predict on unseen test data
predictions = tree.predict(X_test.values)

# Accuracy function
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Evaluate on test data
acc = accuracy(y_test.values, predictions)
print("Test accuracy:", acc)
