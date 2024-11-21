import streamlit as st
import pandas as pd
from preprocessing import load_and_preprocess_data, prepare_single_prediction
from models import ModelTrainer
from visualization import (
    plot_model_comparison, plot_classification_report, 
    plot_confusion_matrix, plot_feature_importance,
    plot_cv_scores, plot_feature_distributions,
    plot_correlation_matrix, plot_feature_vs_target,
    plot_machine_type_distribution, plot_failure_type_distribution
)
import os
import pickle
import plotly.graph_objects as go
from datetime import datetime

def load_raw_data():
    return pd.read_csv('predictive_maintenance.csv')

def load_trained_models():
    models = {}
    results = {}
    model_dir = 'trained models'
    
    if not os.path.exists(model_dir):
        return None, None
        
    try:
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(model_dir, filename)
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                    model_name = filename.replace('.pkl', '')  # Keep the timestamp in name
                    models[model_name] = model_data
                    
                    # Extract results
                    results[model_name] = {
                        'accuracy': model_data.get('accuracy'),
                        'classification_report': model_data.get('classification_report'),
                        'confusion_matrix': model_data.get('confusion_matrix'),
                        'best_params': model_data.get('best_params', {}),
                        'cv_mean_score': model_data.get('cv_results', {}).get('mean_score'),
                        'cv_std_score': model_data.get('cv_results', {}).get('std_score')
                    }
        return models, results
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def main():
    st.title('Predictive Maintenance Application')
    st.sidebar.title('Navigation')
    
    # Initialize session state
    if 'trained_models' not in st.session_state:
        # Try to load pre-trained models
        loaded_models, loaded_results = load_trained_models()
        if loaded_models:
            trainer = ModelTrainer()
            try:
                trainer.trained_models = {}
                trainer.best_params = {}
                trainer.cv_results = {}
                
                for name, data in loaded_models.items():
                    if isinstance(data, dict):
                        trainer.trained_models[name] = data.get('model')
                        trainer.best_params[name] = data.get('best_params', {})
                        trainer.cv_results[name] = data.get('cv_results', {})
                
                # Set best model
                best_accuracy = 0
                trainer.best_model = list(trainer.trained_models.keys())[0]  # Default to first model
                
                for name, data in loaded_models.items():
                    if isinstance(data, dict) and 'accuracy' in data:
                        if data['accuracy'] > best_accuracy:
                            best_accuracy = data['accuracy']
                            trainer.best_model = name
                            trainer.best_accuracy = best_accuracy
                
                st.session_state.trainer = trainer
                st.session_state.results = loaded_results  # Store the loaded results
                st.session_state.trained_models = True
                st.success('Pre-trained models loaded successfully!')
            except Exception as e:
                st.error(f"Error initializing models: {str(e)}")
                st.session_state.trained_models = False
        else:
            st.session_state.trained_models = False
    
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Load scaler for predictions if not already loaded
    if 'scaler' not in st.session_state:
        try:
            _, _, _, _, scaler, _ = load_and_preprocess_data('predictive_maintenance.csv')
            st.session_state.scaler = scaler
        except Exception as e:
            st.error(f"Error loading data preprocessing components: {str(e)}")
    
    # Existing navigation and page selection code would continue here
    page = st.sidebar.radio('Select a Page', 
        ['Data Exploration', 'Model Training', 'Prediction', 'Model Comparison'])
    
    # Implementation of each page would follow based on the previous page logic
    if page == 'Data Exploration':
        # Data exploration page implementation
        df = load_raw_data()
        st.header('Data Exploration')
        
        # Feature distribution plots
        st.subheader('Feature Distributions')
        st.plotly_chart(plot_feature_distributions(df))
        
        # Correlation matrix
        st.subheader('Feature Correlation Matrix')
        st.plotly_chart(plot_correlation_matrix(df))
        
        # Feature vs Target plots
        st.subheader('Feature Distribution by Machine Status')
        feature = st.selectbox('Select Feature', 
            ['Air temperature [K]', 'Process temperature [K]', 
             'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        st.plotly_chart(plot_feature_vs_target(df, feature))
        
        # Machine type and failure type distributions
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Machine Type Distribution')
            st.plotly_chart(plot_machine_type_distribution(df))
        
        with col2:
            st.subheader('Failure Type Distribution')
            st.plotly_chart(plot_failure_type_distribution(df))
    
    elif page == 'Model Training':
        st.header('Model Training')
        
        # Hyperparameter tuning UI
        st.subheader('Hyperparameter Configuration')
        
        with st.expander('Random Forest Parameters'):
            rf_params = {
                'n_estimators': [st.slider('Number of Trees', 50, 500, 200, 50)],
                'max_depth': [st.slider('Maximum Depth', 5, 50, 20, 5)],
                'min_samples_split': [st.slider('Minimum Samples Split', 2, 20, 5, 1)],
                'min_samples_leaf': [st.slider('Minimum Samples Leaf', 1, 10, 2, 1)]
            }
            
        with st.expander('Logistic Regression Parameters'):
            lr_params = {
                'C': [st.select_slider('Regularization Strength', options=[0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)],
                'penalty': ['l2'],  # Limited options for stability
                'solver': ['liblinear']  # Limited options for stability
            }
            
        with st.expander('SVM Parameters'):
            svm_params = {
                'C': [st.select_slider('SVM Regularization Strength', options=[0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)],
                'kernel': [st.selectbox('Kernel', ['rbf', 'linear'], index=0)],
                'gamma': [st.selectbox('Gamma', ['scale', 'auto'], index=0)]  # Removed '0.1' option for stability
            }

        col1, col2 = st.columns(2)
        with col1:
            train_button = st.button('Train Models')
        with col2:
            auto_tune_button = st.button('Auto Tune & Train Models')

        if train_button or auto_tune_button:
            with st.spinner('Training models...'):
                X_train, X_test, y_train, y_test, scaler, le = load_and_preprocess_data('predictive_maintenance.csv')
                
                trainer = ModelTrainer()
                if not auto_tune_button:  # Manual parameters
                    # Use manually configured parameters
                    trainer.models['Random Forest']['params'] = rf_params
                    trainer.models['Logistic Regression']['params'] = lr_params
                    trainer.models['SVM']['params'] = svm_params.copy()
                    if svm_params['gamma'] == '0.1':
                        trainer.models['SVM']['params']['gamma'] = float(svm_params['gamma'])
                else:
                    # Use auto-tuning parameters
                    for model_name in trainer.models:
                        trainer.models[model_name]['params'] = trainer.models[model_name]['auto_params']
                    st.info('Using automated hyperparameter tuning with extended parameter ranges.')
                
                trainer.train_models(X_train, y_train)
                results = trainer.evaluate_models(X_test, y_test)
                
                # Save models and results
                for name, model in trainer.trained_models.items():
                    trainer.save_model(name, model)
                
                st.session_state.trainer = trainer
                st.session_state.results = results
                st.session_state.trained_models = True
                st.success('Models trained successfully!')
        
        if st.session_state.trained_models:
            st.subheader('Training Results')
            st.plotly_chart(plot_model_comparison(st.session_state.results))
            st.plotly_chart(plot_cv_scores(st.session_state.trainer.cv_results))
    
    elif page == 'Prediction':
        st.header('Maintenance Prediction')
        
        if not st.session_state.trained_models:
            st.warning('No trained models available. Please train models first or load existing models.')
        else:
            # Input form for prediction
            st.subheader('Input Machine Parameters')
            
            machine_type = st.selectbox('Machine Type', ['L', 'M', 'H'])
            air_temp = st.number_input('Air Temperature [K]', min_value=290.0, max_value=305.0, value=298.0)
            process_temp = st.number_input('Process Temperature [K]', min_value=300.0, max_value=315.0, value=308.0)
            rotation_speed = st.number_input('Rotational Speed [rpm]', min_value=1000, max_value=3000, value=1500)
            torque = st.number_input('Torque [Nm]', min_value=3.0, max_value=80.0, value=40.0)
            tool_wear = st.number_input('Tool Wear [min]', min_value=0, max_value=250, value=0)
            
            # Model selection
            model_names = list(st.session_state.trainer.trained_models.keys())
            selected_model = st.selectbox('Select Model', model_names, 
                format_func=lambda x: x)  # Remove any formatting/truncating
            
            if st.button('Predict'):
                # Prepare input data
                input_data = {
                    'Type': 0 if machine_type == 'L' else (1 if machine_type == 'M' else 2),
                    'Air temperature [K]': air_temp,
                    'Process temperature [K]': process_temp,
                    'Rotational speed [rpm]': rotation_speed,
                    'Torque [Nm]': torque,
                    'Tool wear [min]': tool_wear
                }
                
                # Make prediction
                X = prepare_single_prediction(input_data, st.session_state.scaler)
                model = st.session_state.trainer.trained_models[selected_model]
                probabilities = st.session_state.trainer.predict(model, X)
                
                # Display results
                st.subheader('Prediction Results')
                st.write(f'Failure Probability: {probabilities[1]:.2%}')
                st.write(f'Normal Operation Probability: {probabilities[0]:.2%}')
    
    elif page == 'Model Comparison':
        st.header('Model Comparison')
        
        if not st.session_state.results:
            st.warning('No model results available. Please train models first or load existing models.')
        else:
            # Overall performance comparison
            st.subheader('Model Performance Comparison')
            st.plotly_chart(plot_model_comparison(st.session_state.results))
            
            # Individual model analysis
            st.subheader('Individual Model Analysis')
            model_name = st.selectbox('Select Model', 
                list(st.session_state.results.keys()),
                format_func=lambda x: x)  # Remove any formatting/truncating
            
            if model_name:
                model_results = st.session_state.results[model_name]
                
                # Classification Report
                st.write('Classification Report')
                st.plotly_chart(plot_classification_report(model_results['classification_report']))
                
                # Confusion Matrix
                st.write('Confusion Matrix')
                st.plotly_chart(plot_confusion_matrix(model_results['confusion_matrix'], model_name))
                
                # Feature Importance
                if model_name in st.session_state.trainer.trained_models:
                    model = st.session_state.trainer.trained_models[model_name]
                    X_train, _, y_train, _, _, _ = load_and_preprocess_data('predictive_maintenance.csv')
                    feature_names = ['Type', 'Air temperature', 'Process temperature', 
                                   'Rotational speed', 'Torque', 'Tool wear']
                    st.plotly_chart(plot_feature_importance(model, feature_names, X_train, y_train))

if __name__ == '__main__':
    main()