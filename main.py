import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from preprocessing import load_and_preprocess_data, prepare_single_prediction
from models import ModelTrainer
from visualization import (
    plot_model_comparison, plot_confusion_matrix, plot_feature_importance, 
    plot_cv_scores, plot_classification_report
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_test_data():
    """Load test data for computing missing metrics"""
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data('predictive_maintenance.csv')
    return X_test, y_test

def load_trained_models():
    """Load all trained models and their results from the 'trained models' directory"""
    models = {}
    results = {}
    
    try:
        # Ensure the trained models directory exists
        os.makedirs('trained models', exist_ok=True)
        model_files = glob.glob(os.path.join('trained models', '*.pkl'))
        
        if not model_files:
            st.info("No pre-trained models found. Please train new models.")
            return None, None
        
        for model_file in model_files:
            try:
                model_name = os.path.basename(model_file).replace('.pkl', '')
                # Skip files with 'best_model' in the name
                if 'best_model' in model_name.lower():
                    continue
                    
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                    if isinstance(model_data, dict) and 'model' in model_data:
                        # Store the model data
                        models[model_name] = model_data
                        
                        # Initialize missing metrics if needed
                        if not all(key in model_data for key in ['confusion_matrix', 'classification_report', 'cv_results']):
                            X_test, y_test = load_test_data()
                            y_pred = model_data['model'].predict(X_test)
                            
                            # Update missing metrics
                            model_data.setdefault('confusion_matrix', 
                                confusion_matrix(y_test, y_pred))
                            model_data.setdefault('classification_report', 
                                classification_report(y_test, y_pred))
                            model_data.setdefault('cv_results', {
                                'mean_score': accuracy_score(y_test, y_pred),
                                'std_score': 0.0
                            })
                            
                            # Save updated model data
                            with open(model_file, 'wb') as f:
                                pickle.dump(model_data, f)
                        
                        # Store results for comparison
                        results[model_name] = {
                            'accuracy': model_data.get('accuracy', 0.0),
                            'classification_report': model_data.get('classification_report', ''),
                            'confusion_matrix': model_data.get('confusion_matrix'),
                            'cv_mean_score': model_data.get('cv_results', {}).get('mean_score', 0.0),
                            'cv_std_score': model_data.get('cv_results', {}).get('std_score', 0.0),
                            'best_params': model_data.get('best_params', {})
                        }
                        
                        st.success(f"Successfully loaded model: {model_name}")
                    else:
                        st.warning(f"Invalid model data format in {model_name}")
                        
            except Exception as e:
                st.warning(f"Error loading model {os.path.basename(model_file)}: {str(e)}")
                continue
        
        if not models:
            st.warning("No valid models could be loaded. Please train new models.")
            return None, None
            
        st.success(f"Successfully loaded {len(models)} models with all required metrics!")
        return models, results
        
    except Exception as e:
        st.error(f"Error accessing models directory: {str(e)}")
        return None, None

def load_raw_data():
    """Load the raw data without preprocessing for exploration"""
    return pd.read_csv('predictive_maintenance.csv')

def plot_feature_distributions(df):
    """Create distribution plots for numerical features"""
    numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    fig = go.Figure()
    for feature in numerical_features:
        fig.add_trace(go.Histogram(
            x=df[feature],
            name=feature,
            nbinsx=30,
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Distribution of Numerical Features',
        barmode='overlay',
        xaxis_title='Value',
        yaxis_title='Count',
        showlegend=True,
        height=500
    )
    return fig

def plot_correlation_matrix(df):
    """Create correlation matrix heatmap"""
    numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target']
    corr_matrix = df[numerical_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=numerical_features,
        y=numerical_features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        width=800
    )
    return fig

def plot_feature_vs_target(df, feature):
    """Create box plots for numerical features vs target"""
    fig = go.Figure()
    
    for target in [0, 1]:
        fig.add_trace(go.Box(
            y=df[df['Target'] == target][feature],
            name=f'{"Failure" if target == 1 else "No Failure"}',
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title=f'{feature} Distribution by Machine Status',
        yaxis_title=feature,
        showlegend=True,
        height=400
    )
    return fig

def plot_machine_type_distribution(df):
    """Create a pie chart for machine type distribution"""
    type_counts = df['Type'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.3
    )])
    
    fig.update_layout(
        title='Distribution of Machine Types',
        height=400
    )
    return fig

def plot_failure_type_distribution(df):
    """Create a bar chart for failure type distribution"""
    failure_counts = df['Failure Type'].value_counts()
    
    fig = go.Figure(data=[go.Bar(
        x=failure_counts.index,
        y=failure_counts.values,
        text=failure_counts.values,
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Distribution of Failure Types',
        xaxis_title='Failure Type',
        yaxis_title='Count',
        height=400
    )
    return fig

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
        
        if st.button('Train Models'):
            with st.spinner('Training models...'):
                X_train, X_test, y_train, y_test, scaler, le = load_and_preprocess_data('predictive_maintenance.csv')
                
                trainer = ModelTrainer()
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
            selected_model = st.selectbox('Select Model', model_names)
            
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
            model_name = st.selectbox('Select Model', list(st.session_state.results.keys()))
            
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