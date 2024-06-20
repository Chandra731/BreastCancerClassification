import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load and Explore the Dataset
cancer = load_breast_cancer()
print(cancer['DESCR'])  # Print dataset description

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print("Features:", df_feat.head().to_markdown(numalign='left', stralign='left'))  # Formatted output
print(f"\nData Types:\n{df_feat.info()}")

# Prepare Data for Modeling
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Initial SVM Model
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n--- Initial SVM Model Results ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100, 1000], 
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, verbose=3)  
grid.fit(X_train, y_train)

# Evaluate Best Model from GridSearch
print("\n--- Best Parameters from Grid Search ---")
print(grid.best_params_)
grid_predictions = grid.predict(X_test)

print("\n--- Best Model Results ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, grid_predictions))
print("Classification Report:\n", classification_report(y_test, grid_predictions))
