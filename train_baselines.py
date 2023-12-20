from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, GridSearchCV
from datasets import load_dataset
import torch
from utils import get_glove_mapping
from nltk.tokenize import word_tokenize


if __name__ == "__main__":
    # Load the LIAR dataset
    dataset = load_dataset("liar")

    # Separate features and labels
    train_texts = [item['statement'] for item in dataset['train']]
    train_labels = [item['label'] for item in dataset['train']]
    valid_texts = [item['statement'] for item in dataset['validation']]
    valid_labels = [item['label'] for item in dataset['validation']]
    test_texts = [item['statement'] for item in dataset['test']]
    test_labels = [item['label'] for item in dataset['test']]

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_texts)
    y_train = train_labels

    # Transform the validation data
    X_val = vectorizer.transform(valid_texts)
    y_val = valid_labels

    # Test 
    X_test = vectorizer.transform(test_texts)
    y_test = test_labels

    # SVM Model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    # Logistic Regression Model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Evaluate SVM Model
    svm_predictions = svm_model.predict(X_val)
    svm_accuracy = accuracy_score(y_val, svm_predictions)
    print("SVM Validation Accuracy:", svm_accuracy)

    # Evaluate Logistic Regression Model
    logistic_predictions = logistic_model.predict(X_val)
    logistic_accuracy = accuracy_score(y_val, logistic_predictions)
    print("Logistic Regression Validation Accuracy:", logistic_accuracy)

    # Attempt 4-fold Validation

    # Combine train and validation texts and labels
    all_texts = dataset['train']['statement'] + dataset['validation']['statement']
    all_labels = dataset['train']['label'] + dataset['validation']['label']

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    all_texts_vectorized = vectorizer.fit_transform(all_texts)

    #Define parameter grid for SVM
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear']
    }

    # Define parameter grid for Logistic Regression
    logistic_param_grid = { 
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'sag', 'saga']
    }

    # Grid search with cross-validation for SVM
    svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=4, scoring='accuracy')
    svm_grid_search.fit(all_texts_vectorized, all_labels)
    best_svm_params = svm_grid_search.best_params_

    # Grid search with cross-validation for Logistic Regression
    logistic_grid_search = GridSearchCV(LogisticRegression(max_iter=1000), logistic_param_grid, cv=4, scoring='accuracy')
    logistic_grid_search.fit(all_texts_vectorized, all_labels)
    best_logistic_params = logistic_grid_search.best_params_

    # Initialize KFold
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # Initialize performance tracking
    svm_accuracies = []
    logistic_accuracies = []

    # Perform 4-fold cross-validation with best parameters
    """
    for train_index, val_index in kf.split(all_texts_vectorized):
        X_train, X_val = all_texts_vectorized[train_index], all_texts_vectorized[val_index]
        y_train, y_val = [all_labels[i] for i in train_index], [all_labels[i] for i in val_index]

        # SVM Model with best parameters
        svm_model = SVC(**best_svm_params)
        svm_model.fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_val)
        svm_accuracies.append(accuracy_score(y_val, svm_predictions))

        # Logistic Regression Model with best parameters
        logistic_model = LogisticRegression(max_iter=1000, **best_logistic_params)
        logistic_model.fit(X_train, y_train)
        logistic_predictions = logistic_model.predict(X_val)
        logistic_accuracies.append(accuracy_score(y_val, logistic_predictions))
    # Calculate and print average accuracies
    print("Average SVM Validation Accuracy:", sum(svm_accuracies) / len(svm_accuracies))
    print("Average Logistic Regression Validation Accuracy:", sum(logistic_accuracies) / len(logistic_accuracies
    ))"""
     # Evaluate SVM Model
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print("SVM Test Accuracy:", svm_accuracy)

    # Evaluate Logistic Regression Model
    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    print("Logistic Regression Test Accuracy:", logistic_accuracy)  