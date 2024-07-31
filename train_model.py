import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Import sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss

def train_model(algorithm, csv_file, sampling_method=None):
    # Load the CSV file for prediction
    df = pd.read_csv(csv_file)

    # Load and concatenate training data
    train_data = pd.concat([
        pd.read_csv(r'D:\ELC_Software_fault_prediction\website\datasets\cm1.csv'),
        pd.read_csv(r'D:\ELC_Software_fault_prediction\website\datasets\jm1.csv'),
        pd.read_csv(r'D:\ELC_Software_fault_prediction\website\datasets\kc3.csv'),
        pd.read_csv(r'D:\ELC_Software_fault_prediction\website\datasets\mc1.csv'),
        pd.read_csv(r'D:\ELC_Software_fault_prediction\website\datasets\mc2.csv')
    ], ignore_index=True)
    
    # Split the training data into features and target
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']

    # Apply sampling techniques if specified
    if sampling_method == 'NearMiss':
        sampler = NearMiss()
    elif sampling_method == 'SMOTE':
        sampler = SMOTE()
    elif sampling_method == 'ADASYN':
        sampler = ADASYN()
    elif sampling_method == 'SMOTE+ENN':
        sampler = SMOTEENN()
    else:
        sampler = None

    if sampler:
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    # Initialize the model
    if algorithm == 'Random Forest':
        model = RandomForestClassifier()
    elif algorithm == 'Naive Bayes':
        model = GaussianNB()
    elif algorithm == 'SVM':
        model = SVC(probability=True)  # probability=True to enable roc_auc_score
    elif algorithm == 'Logistic Regression':
        model = LogisticRegression()
    elif algorithm == 'KNN':
        model = KNeighborsClassifier()
    elif algorithm == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif algorithm == 'Gradient Boosting':
        model = GradientBoostingClassifier()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Train the model
    model.fit(X_train, y_train)

    # Prepare the test data for prediction
    X_test = df.drop('label', axis=1, errors='ignore')  # Ensure 'label' is not present in test data

    # Ensure the test data has the same features as training data
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Calculate performance metrics
    y_test = df.get('label')  # Test labels might be missing; handle if necessary
    if y_test is not None and len(y_test) == len(X_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # For classifiers with predict_proba
    else:
        accuracy = precision = recall = f1 = auc = None
        print("Warning: Test data labels are missing or do not match the number of samples.")

    # Generate visualizations
    plt.figure(figsize=(10, 5))
    plt.hist(y_pred, bins=[0, 1, 2], alpha=0.7, label='Predicted Classes')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Classes')
    plt.legend()
    plt.show()

    return accuracy, precision, recall, f1, auc
