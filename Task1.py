
# ========================================
# TASK 1: Credit Scoring Model
# ========================================


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def credit_scoring_model():
    """Simple Credit Scoring Model using Random Forest"""
    
    # Create sample data (replace with real dataset)
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'income': np.random.randint(20000, 100000, n_samples),
        'debt': np.random.randint(0, 50000, n_samples),
        'credit_history': np.random.randint(1, 10, n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples)
    }
    
    # Create target variable (good/bad credit)
    df = pd.DataFrame(data)
    df['credit_score'] = ((df['income'] > 40000) & 
                         (df['debt'] < 20000) & 
                         (df['credit_history'] > 5)).astype(int)
    
    print("Credit Scoring Model")
    print("=" * 50)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['credit_score'].value_counts()}")
    
    # Features and target
    X = df.drop('credit_score', axis=1)
    y = df['credit_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, accuracy