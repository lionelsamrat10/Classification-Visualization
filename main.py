# Author: Samrat Mitra
# Github Link: https://github.com/lionelsamrat10

# The datasets are used from UC Irvine Machine Learning Repository
# These three datsets are already available at sklearn

# Import the libraries
import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Importing the Classifiers 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Importing the function that Splits the data into training and test set 
from sklearn.model_selection import train_test_split

# Find the accuracy of the Classifiers
from sklearn.metrics import accuracy_score

# To perform Dimensionality Reduction using PCA (Principal Component Analysis)
from sklearn.decomposition import PCA

# Create the heading and the description
st.title("Classification Algorithms, applied to classic datasets")
st.write("""
    # Explore different Classifier
    Which one is the best?
""")

# Create Dropdown to select the dataset
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

# Print the name of the dataset
st.write(f"## {dataset_name} Dataset")

# Create Dropdown to select the Classifier
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

#Function to load the dataset
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    # Split the data into X and y
    # X contains the independent variables
    # y contains the dependent variables
    X = data.data
    y = data.target
    return X, y

# Get the dataset and printing its details
X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

# Based on the Classifier name this function prints different values in our UI
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15) # Here K denotes number of neighbours in our classifier
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0) # C value lies in this range here (0.01 to 10)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15) # max_depth denotes Maximum depth of the trees in our classifier
        n_estimators = st.sidebar.slider("Number_Of_Estimtors", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

# Get the required parameters to be passed to the Classifiers
params = add_parameter_ui(classifier_name)

# Create the Classfiers
def get_classifier(clf_name, params):
    if clf_name == "KNN": # takes number of neighbours as the parameter
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_name == "SVM": # Takes C as the parameter
        clf = SVC(C =  params["C"])
    else: # RandomForestClassifier takes number of estimators and max_depth of the decision trees as param
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                     max_depth = params["max_depth"], random_state = 1234)
    return clf

# clf contains the selected classifier type
clf = get_classifier(classifier_name, params)

# Perform the Classification
# Step - 01: Splitting the dataset into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# Step - 02: Training the Classifier using the training set
clf.fit(X_train, y_train)

# Step - 03: Our Classifier makes its predictions against the Test dataset
y_pred = clf.predict(X_test)

# Step - 04: Finding the accuracy of our model
accuracy = accuracy_score(y_test, y_pred) * 100

# Printing the results
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {accuracy} %")

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
# We are using PCA here to reduce the dimension of our data to 2
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
st.write("""
    ### Visualizing the dataset
""")
plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = "viridis")
plt.title(f"{dataset_name} Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# alternative of plt.show in Streamlit is st.pyplot() 
st.pyplot(fig)