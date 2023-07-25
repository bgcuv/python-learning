import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    win_size = (64, 128)
    block_size = (32, 32)
    block_stride = (8, 8)
    cell_size = (16, 16)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    resized_image = cv2.resize(gray_image, (640, 480))
    hog_features = hog.compute(resized_image)

    return hog_features


def load_dataset(data_directory):
    xml_files = glob.glob(os.path.join(data_directory, "*.xml"))

    X = []
    y = []

    for xml_file in xml_files:
        image_file = (
            os.path.splitext(xml_file)[0] + ".jpg"
        )  # Assuming the image files are in .jpg format

        tree = ET.parse(xml_file)
        root = tree.getroot()
        label = root.find("object").find("name").text

        image = cv2.imread(image_file)
        if image is None:
            continue

        hog_features = extract_hog_features(image)
        X.append(hog_features)
        y.append(label)

    return np.array(X), np.array(y)


data_directory = "export"
X, y = load_dataset(data_directory)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid_forest = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# knn_classifier = KNeighborsClassifier(n_neighbors=3)
# knn_classifier.fit(X_train, y_train)

# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(
#     estimator=rf, param_grid=param_grid_forest, cv=5, verbose=2, n_jobs=-1
# )
# grid_search.fit(X_train, y_train)
# print("Best parameters found: ", grid_search.best_params_)
# best_rf = grid_search.best_estimator_

# print("Logistic Regression")
# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)
# y_pred_log_reg = log_reg.predict(X_test)

# print(classification_report(y_test, y_pred_log_reg))

print("SGD Classifier")
sgd_clf = SGDClassifier()
for i in range(20):
    subset = np.random.choice(len(X_train), 1000)
    sgd_clf.partial_fit(X_train[subset], y_train[subset], classes=np.unique(y_train))

y_pred_sgd_clf = sgd_clf.predict(X_test)

print(classification_report(y_test, y_pred_sgd_clf))

# y_pred_knn = knn_classifier.predict(X_test)
# y_pred_forest = best_rf.predict(X_test)

# print("KNN")
# print(classification_report(y_test, y_pred_knn))
# print("Random Forest")
# print(classification_report(y_test, y_pred_forest))
