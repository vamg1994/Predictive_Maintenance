from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import pickle
import os
import streamlit as st
import pandas as pd

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'auto_params': {
                    'n_estimators': [50, 100, 200, 300, 400],
                    'max_depth': [5, 10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 4, 5, 8, 10],
                    'min_samples_leaf': [1, 2, 3, 4]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'auto_params': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.1]
                },
                'auto_params': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
                }
            }
        }
        self.trained_models = {}
        self.best_model = None
        self.best_accuracy = 0
        self.best_params = {}
        self.cv_results = {}
        self.X_test = None
        self.y_test = None

    def perform_cross_validation(self, X, y, n_folds=5):
        """Perform k-fold cross-validation for each model"""
        cv_results = {}
        for name, model_info in self.models.items():
            # Create a new instance with default parameters
            model = model_info['model']
            scores = cross_val_score(model, X, y, cv=n_folds, scoring='accuracy')
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'all_scores': scores
            }
        self.cv_results = cv_results
        return cv_results

    def train_models(self, X_train, y_train):
        # First perform cross-validation
        self.perform_cross_validation(X_train, y_train)
        
        # Then proceed with GridSearchCV and model training
        for name, model_info in self.models.items():
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=5,
                n_jobs=-1,
                scoring='accuracy',
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.trained_models[name] = grid_search.best_estimator_
            self.best_params[name] = grid_search.best_params_

    def evaluate_models(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'best_params': self.best_params[name],
                'cv_mean_score': self.cv_results[name]['mean_score'],
                'cv_std_score': self.cv_results[name]['std_score']
            }
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = name

        return results

    def save_model(self, model_name, model):
        try:
            os.makedirs('trained models', exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join('trained models', f'{model_name}_{timestamp}.pkl')
            
            # Calculate metrics if X_test and y_test are available
            accuracy = None
            classification_report_text = None
            conf_matrix = None
            
            if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                classification_report_text = classification_report(self.y_test, y_pred)
                conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            model_data = {
                'model': model,
                'best_params': self.best_params.get(model_name, {}),
                'cv_results': self.cv_results.get(model_name, {}),
                'accuracy': accuracy,
                'classification_report': classification_report_text,
                'confusion_matrix': conf_matrix
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            st.error(f"Error saving model {model_name}: {str(e)}")

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            return model_data['model']

    def predict(self, model, X):
        return model.predict_proba(X)[0]
