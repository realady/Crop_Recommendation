# -*- coding: utf-8 -*-
"""Crop Recommendation Using Machine Learning"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Display the first few rows
print(crop.head())

# Display dataset shape
print(crop.shape)

# Display dataset info
crop.info()

# Check for null values
print(crop.isnull().sum())

# Check for duplicates
print(crop.duplicated().sum())

# Display statistical summary
print(crop.describe())

# Create a dictionary to map crop labels to numerical values
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6, 'orange': 7,
    'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18,
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Map the labels to numerical values
crop['label'] = crop['label'].map(crop_dict)

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(crop.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Display value counts of the 'label' column
print(crop.label.value_counts())

# Display unique count of 'label' column
print(crop['label'].unique().size)

# Distribution plot for 'P'
plt.figure()
sns.distplot(crop['P'])
plt.title('Distribution of P')
plt.show()

# Distribution plot for 'N'
plt.figure()
sns.distplot(crop['N'])
plt.title('Distribution of N')
plt.show()

# Display unique labels
print(crop.label.unique())

# Display the first few rows after mapping
print(crop.head())

# Display value counts after mapping
print(crop.label.value_counts())

# Split the data into features and target
X = crop.drop('label', axis=1)
y = crop['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply MinMax scaling
mx = MinMaxScaler()
X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)

# Apply Standard scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define models
models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'ExtraTreeClassifier': ExtraTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} model with accuracy: {score}")

# Train and evaluate Random Forest Classifier
randclf = RandomForestClassifier()
randclf.fit(X_train, y_train)
y_pred = randclf.predict(X_test)
print(f"Random Forest Classifier accuracy: {accuracy_score(y_test, y_pred)}")

# Define recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_mx_features = sc.transform(mx_features)
    prediction = randclf.predict(sc_mx_features).reshape(1, -1)
    return prediction[0]

# Test the recommendation function
N, P, K, temperature, humidity, ph, rainfall = 90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536
predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print(f"Predicted crop: {list(crop_dict.keys())[list(crop_dict.values()).index(predict)]}")

# Save the model and scalers
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

print("Model and scalers saved successfully.")