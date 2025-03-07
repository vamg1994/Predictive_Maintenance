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
    
    # Add sidebar with contact information and rankings
    with st.sidebar:
        st.markdown("### Virgilio Madrid - Data Scientist")
        st.link_button("Portfolio 🌐", "https://portfolio-vam.vercel.app/")
        st.link_button("LinkedIn 💼", "https://www.linkedin.com/in/vamadrid/")
        st.link_button("E-Mail 📧", "mailto:virgiliomadrid1994@gmail.com")
        
        # Add model rankings if models are trained
        if 'trained_models' in st.session_state and st.session_state.trained_models:
            st.subheader("Top 3 Models")
            results = st.session_state.results
            if results:
                # Sort models by accuracy and take top 3
                sorted_models = sorted(
                    results.items(), 
                    key=lambda x: x[1]['accuracy'], 
                    reverse=True
                )[:3]  # Only take top 3
                for model_name, model_data in sorted_models:
                    accuracy = model_data['accuracy']
                    st.metric(model_name, f"{accuracy:.2%} accuracy")
        else:
            st.info("No trained models available")
    
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

    # Replace radio buttons with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Exploration", 
        "🔧 Model Training", 
        "🎯 Prediction", 
        "📈 Model Comparison",
        "❓ FAQ & Technical Details"
    ])
    
    # Data Exploration tab
    with tab1:
        df = load_raw_data()
        st.header('Data Exploration')
        
        # Feature distribution plots
        st.subheader('Feature Distributions')
        st.plotly_chart(plot_feature_distributions(df))
        st.markdown("""
        **Feature Distribution Interpretation:**
        This plot shows the distribution of each feature in the dataset.
        X-axis: Feature values
        Y-axis: Frequency of each feature value \n
        **Feature Names:**
        - Air temperature: Temperature of the air in the machine.
        - Process temperature: Temperature of the process in the machine.
        - Rotational speed: Speed of the machine.
        - Torque: Torque of the machine.
        - Tool wear: Wear of the tool in the machine.
        """)
        
        # Correlation matrix
        st.subheader('Feature Correlation Matrix')
        st.plotly_chart(plot_correlation_matrix(df))
        st.markdown("""
        **Correlation Matrix Interpretation:**
        - **Positive Correlation:** Features tend to increase or decrease together.
        - **Negative Correlation:** Features tend to move in opposite directions.
        - **Strong Correlation:** Values closer to 1 or -1.
        - **Weak Correlation:** Values closer to 0.
        """)
        
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
            st.markdown("""
            **Machine Type Distribution Interpretation:**
            This distribution(raw data) shows the percentage of machines for each machine type.
            """)
        
        with col2:
            st.subheader('Failure Type Distribution')
            st.plotly_chart(plot_failure_type_distribution(df))
            st.markdown("""
            **Failure Type Distribution Interpretation:**
            This distribution(raw data) shows the percentage of failures for each machine type.
            In data engineering, it was necessary to balance target classes to avoid bias.
            """)
    
    # Model Training tab
    with tab2:
        st.header('Model Training')
        
        # Keep existing model training code
        with st.expander('Hyperparameter Configuration'):
            # Random Forest Parameters
            st.subheader('Random Forest Parameters')
            rf_params = {
                'n_estimators': [st.slider('Number of Trees', 50, 500, 200, 50)],
                'max_depth': [st.slider('Maximum Depth', 5, 50, 20, 5)],
                'min_samples_split': [st.slider('Minimum Samples Split', 2, 20, 5, 1)],
                'min_samples_leaf': [st.slider('Minimum Samples Leaf', 1, 10, 2, 1)]
            }
            
            # Logistic Regression Parameters
            st.subheader('Logistic Regression Parameters')
            lr_params = {
                'C': [st.select_slider('Regularization Strength', options=[0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
            
            # SVM Parameters
            st.subheader('SVM Parameters')
            svm_params = {
                'C': [st.select_slider('SVM Regularization Strength', options=[0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)],
                'kernel': [st.selectbox('Kernel', ['rbf', 'linear'], index=0)],
                'gamma': [st.selectbox('Gamma', ['scale', 'auto'], index=0)]
            }

        col1, col2 = st.columns(2)
        with col1:
            train_button = st.button('Train Models')
        with col2:
            auto_tune_button = st.button('Auto Tune & Train Models')

        st.markdown("""
        **Model Training Interpretation:**
        This section allows you to train the models with the parameters you have configured.
        You can also use the auto tune button to automatically tune the parameters and train the models.
        """)
        
        # Rest of the existing training code
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
            st.markdown("""
 #### **Cross-Validation**
- **Purpose**: Evaluates model performance during training and helps fine-tune hyperparameters or select the best model.
- **How It Works**: 
  - The dataset is split into multiple folds (e.g., 5 or 10). 
  - The model is trained on some folds and validated on the remaining fold. 
  - This process is repeated for each fold, and results are averaged.
- **Key Point**: Performed on the **training dataset** only; the test set is not used at this stage.
- **Benefit**: Reduces overfitting by providing a robust evaluation on unseen splits of the training data.

#### **Test Set**
- **Purpose**: Evaluates the final model's performance on unseen, untouched data.
- **How It Works**: 
  - After all training and validation (including cross-validation), the final model is tested on the separate test set. 
  - The test set provides a proxy for future data.
- **Key Point**: The test set must remain **completely separate** from training and validation to ensure unbiased evaluation.
- **Benefit**: Gives an accurate estimate of the model’s real-world generalization.

---

### **Key Differences**
| Aspect              | Cross-Validation                          | Test Set                                  |
|---------------------|-------------------------------------------|------------------------------------------|
| **When Used**       | During training                           | After training and tuning                |
| **Purpose**         | Model selection and hyperparameter tuning | Final model evaluation                   |
| **Data Used**       | Training dataset (split into folds)       | Separate, unseen dataset                 |
| **Goal**            | Reduce overfitting and improve robustness | Assess real-world performance            |

---

**Analogy**:  
Cross-validation is like practice exams to prepare for the final test. The test set is the actual exam that determines how well the model performs in the real world.
            """)
    
    # Prediction tab
    with tab3:
        st.header('Maintenance Prediction')
        
        if not st.session_state.trained_models:
            st.warning('No trained models available. Please train models first or load existing models.')
        else:
            # Keep existing prediction code
            st.subheader('Input Machine Parameters')
            st.markdown("""
            **Input Machine Parameters Interpretation:**
            This section allows you to input the machine parameters for prediction.
            The output will be the probability of failure or normal operation.
            You can select model from the dropdown menu.
            """)
            
            
            col1, col2 = st.columns(2)
            with col1:
                machine_type = st.selectbox('Machine Type', ['L', 'M', 'H'])
                air_temp = st.number_input('Air Temperature [K]', min_value=290.0, max_value=305.0, value=298.0)
                process_temp = st.number_input('Process Temperature [K]', min_value=300.0, max_value=315.0, value=308.0)
            
            with col2:
                rotation_speed = st.number_input('Rotational Speed [rpm]', min_value=1000, max_value=3000, value=1500)
                torque = st.number_input('Torque [Nm]', min_value=3.0, max_value=80.0, value=40.0)
                tool_wear = st.number_input('Tool Wear [min]', min_value=0, max_value=250, value=0)
            
            # Model selection and prediction
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
    
    # Model Comparison tab
    with tab4:
        st.header('Model Comparison')
        
        if not st.session_state.results:
            st.warning('No model results available. Please train models first or load existing models.')
        else:
            # Add unique key to the first model comparison plot
            st.subheader('Model Performance Comparison')
            st.plotly_chart(plot_model_comparison(st.session_state.results), key='model_comparison_tab4')
            
            st.subheader('Individual Model Analysis')
            model_name = st.selectbox('Select Model for Detailed Analysis', 
                list(st.session_state.results.keys()))
            
            if model_name:
                model_results = st.session_state.results[model_name]
                
                # Add unique keys to all plotly charts
                st.write('Classification Report')
                st.plotly_chart(plot_classification_report(model_results['classification_report']), 
                              key=f'classification_report_{model_name}')
                st.markdown("""
#### **1. Precision**
- **Meaning**: Proportion of correct positive predictions out of all positive predictions.
- **Formula**: True Positives / (True Positives + False Positives)

#### **2. Recall**
- **Meaning**: Proportion of actual positives correctly identified.
- **Formula**: True Positives / (True Positives + False Negatives)

#### **3. F1-Score**
- **Meaning**: Harmonic mean of Precision and Recall.
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)

#### **4. Support**
- **Meaning**: Number of true instances of each class in the dataset.
                """)
                
                st.write('Confusion Matrix')
                st.plotly_chart(plot_confusion_matrix(model_results['confusion_matrix'], model_name), 
                              key=f'confusion_matrix_{model_name}')
                st.markdown("""
                **Confusion Matrix Interpretation:**
                - **True Positives (TP)**: Correctly predicted as positive.
                - **True Negatives (TN)**: Correctly predicted as negative.
                - **False Positives (FP)**: Incorrectly predicted as positive.
                - **False Negatives (FN)**: Incorrectly predicted as negative.
                """)
                
                if model_name in st.session_state.trainer.trained_models:
                    model = st.session_state.trainer.trained_models[model_name]
                    X_train, _, y_train, _, _, _ = load_and_preprocess_data('predictive_maintenance.csv')
                    feature_names = ['Type', 'Air temperature', 'Process temperature', 
                                   'Rotational speed', 'Torque', 'Tool wear']
                    st.plotly_chart(plot_feature_importance(model, feature_names, X_train, y_train),
                                  key=f'feature_importance_{model_name}')
                    st.markdown("""
                    **Feature Importance Interpretation:**
                    This plot shows the importance of each feature in the model.
                    """)

    # FAQ tab
    with tab5:
        
        st.header("Technical Documentation & FAQ")
        
        st.markdown("""
        **Technical Documentation & FAQ Interpretation:**
        This section provides technical details and answers to common questions about the application.
        
        **What is the purpose of this project?**
        This project is a predictive maintenance application that uses machine learning to predict the probability of failure or normal operation of a machine.
        This kind of application can be used in many industries such as aerospace, automotive, manufacturing, etc.
        Also, machine learning may help us predict customer churn, fraud detection, etc.
        """)
        
        st.subheader("🔧 Tech Stack")
        st.markdown("""
        **Frontend:**
        - Streamlit: Real-time web application framework
        - Plotly: Interactive visualization library
        
        **Backend:**
        - Python
        - Scikit-learn: ML model implementation and evaluation
        - Pandas: Data manipulation and preprocessing
        - NumPy: Numerical computations
        
        **Model Persistence:**
        - Pickle: Model serialization
        - Local file system for model storage
        
        **Development Tools:**
        - Git: Version control
        - Virtual Environment: Dependency isolation
        """)

        st.subheader("🔄 Application Workflow")
        
        with st.expander("📊 Data Exploration Tab"):
            st.markdown("""
            **Purpose:** Data analysis and visualization module
            
            **Key Features:**
            - Feature distribution analysis using Plotly histograms
            - Correlation matrix visualization using Plotly heatmaps
            - Feature-target relationship analysis
            - Machine type and failure distribution insights
            
            **Technical Implementation:**
            - Pandas for data manipulation
            - Plotly for interactive visualizations
            - Visualization functions in `visualization.py`
            """)
            
        with st.expander("🔧 Model Training Tab"):
            st.markdown("""
            **Purpose:** Model training and hyperparameter optimization interface
            
            **Key Features:**
            - Implementation of multiple ML algorithms:
                - Random Forest Classifier
                - Logistic Regression
                - Support Vector Machine
            - GridSearchCV for hyperparameter optimization
            - Cross-validation for model validation
            
            **Technical Implementation:**
            - Scikit-learn's model implementations
            - ModelTrainer class in `models.py`
            - Manual and automated hyperparameter tuning options
            """)
            
        with st.expander("🎯 Prediction Tab"):
            st.markdown("""
            **Purpose:** Real-time machine failure prediction interface
            
            **Key Features:**
            - Input interface for machine parameters
            - Real-time predictions using trained models
            - Probability scores for failure prediction
            
            **Technical Implementation:**
            - StandardScaler for feature scaling
            - Model inference using trained models
            - Streamlit session state for model persistence
            """)
            
        with st.expander("📈 Model Comparison Tab"):
            st.markdown("""
            **Purpose:** Model evaluation and comparison
            
            **Key Features:**
            - Performance metrics visualization
            - Classification reports
            - Confusion matrices
            - Feature importance analysis (for supported models)
            
            **Technical Implementation:**
            - Scikit-learn's evaluation metrics
            - Plotly visualization functions
            - Dynamic model selection
            """)
            
        st.subheader("🔍 Technical Implementation Details")
        
        with st.expander("Model Architecture & Training"):
            st.markdown("""
            **Data Pipeline:**
            1. Data Preprocessing:
               - Standard scaling for numerical features
               - Label encoding for categorical variables
               - Train-test split with stratification
            
            **Training Process:**
            1. Cross-validation (k=5 folds)
            2. GridSearchCV for hyperparameter optimization
            3. Model evaluation on hold-out test set
            
            **Model Storage:**
            - Models saved with:
                - Training timestamp
                - Performance metrics
                - Hyperparameters
                - Cross-validation results
            """)
            
        with st.expander("Performance Considerations"):
            st.markdown("""
            **Current Implementations:**
            - Basic data loading with Pandas
            - GridSearchCV parallel processing
            - Session state for model persistence
            - Streamlit caching for data loading
            
            **Memory Handling:**
            - Model loading on demand
            - Basic preprocessing pipeline
            """)
            
        with st.expander("Code Organization"):
            st.markdown("""
            **Project Structure:**
            - Modular architecture with separate files for:
                - Data preprocessing (`preprocessing.py`)
                - Model training (`models.py`)
                - Visualization (`visualization.py`)
                - Main application (`main.py`)
            
            **Error Handling:**
            - Basic error handling for:
                - Data loading
                - Model training
                - Prediction operations
            """)

if __name__ == '__main__':
    main()