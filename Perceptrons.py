import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Perceptron

# Loading and preprocessing the data
df = pd.read_csv('dataset.csv')
df = df.rename(columns={'Unnamed: 0': 'targets'})
labelEncoder = LabelEncoder()
df['targets'] = df['targets'].apply(lambda x: re.sub(r'\d+', '', x))
df['targets'] = labelEncoder.fit_transform(df['targets'])
df = df.dropna()

# Feature scaling
scaler = StandardScaler()
X = df.drop(columns=['targets'])
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
Y = df['targets']

# Feature selection
X = X.loc[:, X.nunique() > 1]

# Handle imbalanced data in order to resample 
smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X, Y)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=42)

# kNN and its parameter grid
knn_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': range(1, 50, 2),  # Odd numbers
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'n_jobs': [-1]
}

# Using GridSearchCV to find k
grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)  
grid_search.fit(X_train, Y_train)

# k value
optimal_k = grid_search.best_params_['n_neighbors']
print(f"Optimal k value: {optimal_k}")

# estimation
estimation = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(estimation, X_res, Y_res, cv=3, n_jobs=-1)  
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean()}")

# Evaluating on test set
Y_predict = estimation.predict(X_test)
print(f"Test accuracy: {accuracy_score(Y_test, Y_predict)}")

# Evaluating with optimal k
Y_predict_knn = estimation.predict(X_test)
print(f"kNN Test accuracy with optimal k: {accuracy_score(Y_test, Y_predict_knn)}")

results = grid_search.cv_results_
plt.plot(range(1, 50, 2), results['mean_test_score'][:len(range(1, 50, 2))], marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('kNN Accuracy vs. k')
plt.show()

# Perceptron model
percep_model = Perceptron(random_state=42, n_jobs=-1)

# Train the Perceptron
percep_model.fit(X_train, Y_train)

# Cross-validation for Perceptron
perceptron_cv_scores = cross_val_score(percep_model, X_res, Y_res, cv=3, n_jobs=-1)
print(f"Perceptron Cross-validation scores: {perceptron_cv_scores}")
print(f"Mean Perceptron CV accuracy: {perceptron_cv_scores.mean()}")

# Evaluate on the test set
Y_pred_perceptron = percep_model.predict(X_test)
print(f"Perceptron Test accuracy: {accuracy_score(Y_test, Y_pred_perceptron)}")
print("\nPerceptron Classification Report:")
print(classification_report(Y_test, Y_pred_perceptron))

