import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import joblib

def load_and_clean_data(data_path):
    data = pd.read_csv(data_path).iloc[:, :-1]
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        data[column] = data[column].fillna(data[column].mean())
    return data

def load_labels(labels_path):
    labels = pd.read_csv(labels_path)
    return labels.to_numpy().ravel()

def split_data(data, labels, test_size=0.2, random_state=42):
    return train_test_split(data, labels, test_size=test_size, random_state=random_state)

def select_features(data_train, labels_train, data_test, k=17):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(data_train, labels_train)
    return selector.transform(data_train), selector.transform(data_test)

def save_data(data, path):
    pd.DataFrame(data).to_csv(path, index=False)

def train_and_evaluate(data_train, labels_train, data_test, labels_test):
    model = GaussianNB(var_smoothing=1e-09)
    model.fit(data_train, labels_train)
    labels_pred = model.predict(data_test)
    accuracy = f1_score(labels_test, labels_pred, average='weighted')
    return model, accuracy, labels_pred

def main():
    data = load_and_clean_data("traindata.txt")
    labels = load_labels("trainlabels.txt")
    data_arr = data.to_numpy()
    data_train, data_test, labels_train, labels_test = split_data(data_arr, labels)
    data_train_new, data_test_new = select_features(data_train, labels_train, data_test, k=17)
    save_data(data_train, "data_train.csv")
    save_data(labels_train, "labels_train.csv")
    save_data(data_test, "data_test.csv")
    save_data(labels_test, "labels_test.csv")
    model, f1, _ = train_and_evaluate(data_train_new, labels_train, data_test_new, labels_test)
    # Save the trained model for later use
    joblib.dump(model, "rail_model.joblib")
    print(f"Weighted F1 Score: {f1:.4f}")
if __name__ == "__main__":
    main()