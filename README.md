# ⚡ Perceptron Classifier on Genetic Data

This pipeline explores the power of a simple, classic neural network — the Perceptron — in classifying mutation types from genetic data. It’s like giving your dataset a basic brain and watching it learn on its own.

🚀 What does this pipeline do?
Cleans and labels the target column with some regex magic and LabelEncoder.

Scales features using StandardScaler — because raw features can be unruly.

Removes low-variance features that don’t contribute anything meaningful.

Fights imbalance with SMOTE, ensuring all mutation classes get a fair shot.

Splits the dataset for training and testing — the classic 80/20 way.

Trains a Perceptron model and cross-validates it using 3-fold CV.

Evaluates both cross-validation scores and test set accuracy.

Prints out a detailed classification report to show precision, recall, and F1-scores across all classes.

