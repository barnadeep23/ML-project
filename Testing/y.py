import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_diabetes_model():
    diabetes_data = pd.read_csv('diabetes.csv')
    X = diabetes_data.drop(columns='Outcome', axis=1)
    Y = diabetes_data['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = svm.SVC(kernel='linear')
    model.fit(X_train, Y_train)
    
    print("Black Box Testing: Diabetes Prediction")
    test_input = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
    prediction = model.predict(test_input)
    print("Prediction:", "Diabetes Detected" if prediction[0] == 1 else "No Diabetes")
    
    print("White Box Testing: Diabetes Prediction")
    print("Model Accuracy:", accuracy_score(Y_test, model.predict(X_test)))
    edge_case_input = np.array([[1, 10000, 66, 29, 0, 26.6, 0.351, 31]])
    print("Edge Case Prediction:", "Handled" if model.predict(edge_case_input)[0] == 1 else "Not Handled")

def test_heart_model():
    heart_data = pd.read_csv('heart.csv')
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    
    print("Black Box Testing: Heart Disease Prediction")
    test_input = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    prediction = model.predict(test_input)
    print("Prediction:", "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")
    
    print("White Box Testing: Heart Disease Prediction")
    print("Model Accuracy:", accuracy_score(Y_test, model.predict(X_test)))
    print("Decision Path:", model.decision_function(test_input))

def test_parkinsons_model():
    parkinsons_data = pd.read_csv('parkinsons.csv')
    X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
    Y = parkinsons_data['status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = svm.SVC(kernel='rbf')
    model.fit(X_train, Y_train)
    
    print("Black Box Testing: Parkinson's Disease Prediction")
    test_input = np.array([[197.144, 218.664, 197.144, 0.00556, 0.00003, 0.00268, 0.00335, 0.01267, 0.02211, 
                            0.00639, 0.01078, 0.00468, 0.19562, 0.01133, 0.01607, 0.01384, 0.02086, 0.00627, 
                            22.423, 0.42857, 0.86555]])
    prediction = model.predict(test_input)
    print("Prediction:", "Parkinson's Detected" if prediction[0] == 1 else "No Parkinson's")
    
    print("White Box Testing: Parkinson's Disease Prediction")
    print("Model Accuracy:", accuracy_score(Y_test, model.predict(X_test)))
    print("Kernel Evaluation:", model.decision_function(test_input))

# Run all tests
test_diabetes_model()
test_heart_model()
test_parkinsons_model()

