import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Dataset yuklash
df = pd.read_csv('data/heart.csv')

# X va Y ajratish
X = df.drop('target', axis=1)
y = df['target']

# Train/Test bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3 ta model
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

print("=" * 50)
print("MODELLAR NATIJALARI:")
print("=" * 50)

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name}: {score*100:.2f}% aniqlik")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\nEng yaxshi model: {best_name} ({best_score*100:.2f}%)")

# Eng yaxshi modelni saqlash
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\nModel saqlandi: model.pkl - OK!")