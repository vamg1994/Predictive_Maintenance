import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

def plot_model_comparison(results):
    # Create a DataFrame for comparison
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'CV Mean Accuracy': [results[model]['cv_mean_score'] for model in results.keys()],
        'CV Std': [results[model]['cv_std_score'] for model in results.keys()]
    })
    
    # Create bar plot with error bars
    fig = go.Figure()
    
    # Add test accuracy bars
    fig.add_trace(go.Bar(
        name='Test Accuracy',
        x=df['Model'],
        y=df['Test Accuracy'],
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add CV accuracy bars with error bars
    fig.add_trace(go.Bar(
        name='CV Mean Accuracy',
        x=df['Model'],
        y=df['CV Mean Accuracy'],
        error_y=dict(
            type='data',
            array=df['CV Std'],
            visible=True
        ),
        marker_color='green',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Model Performance Comparison (Test vs Cross-Validation)',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        barmode='group',
        yaxis_range=[0, 1]
    )
    
    return fig

def plot_classification_report(report_text):
    """
    Convert the classification report text into an interactive table visualization.
    
    Args:
        report_text: String containing sklearn's classification report
    
    Returns:
        Plotly figure object containing the styled table
    """
    # Parse the text report into a DataFrame
    report_lines = report_text.split('\n')
    report_data = []
    
    for line in report_lines[2:-3]:  # Skip header and footer
        if line.strip():
            row = line.strip().split()
            if len(row) >= 5:  # Valid data row
                class_name = row[0]
                precision = float(row[1])
                recall = float(row[2])
                f1_score = float(row[3])
                support = int(row[4])
                report_data.append([class_name, precision, recall, f1_score, support])
    
    df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    # Create an enhanced table with better formatting and colors
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Class</b>', '<b>Precision</b>', '<b>Recall</b>', 
                   '<b>F1-Score</b>', '<b>Support</b>'],
            fill_color='#2c3e50',
            align=['left', 'center', 'center', 'center', 'center'],
            font=dict(color='white', size=14),
            height=40
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[
                ['#f8f9fa'] * len(df),  # Class column
                [f'rgba(46, 204, 113, {v})' for v in df['Precision']],  # Precision
                [f'rgba(52, 152, 219, {v})' for v in df['Recall']],     # Recall
                [f'rgba(155, 89, 182, {v})' for v in df['F1-Score']],   # F1-Score
                ['#f8f9fa'] * len(df)    # Support column
            ],
            align=['left', 'center', 'center', 'center', 'center'],
            font=dict(color=['black'] * 5, size=13),
            height=35,
            format=[
                None,              # Class
                '.3%',            # Precision
                '.3%',            # Recall
                '.3%',            # F1-Score
                'd',             # Support
            ]
        )
    )])
    
    fig.update_layout(
        title=dict(
            text='Classification Report',
            x=0.5,
            font=dict(size=18)
        ),
        margin=dict(t=50, b=20, l=0, r=0),
        height=max(200, 100 + (len(df) * 35)),
        width=800
    )
    
    return fig

def plot_confusion_matrix(conf_matrix, model_name):
    if conf_matrix is None:
        # Create a figure with a message when no confusion matrix is available
        fig = go.Figure()
        fig.add_annotation(
            text="No confusion matrix data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            width=700,
            height=600
        )
        return fig

    # Calculate percentages for annotations
    total = conf_matrix.sum()
    percentage_matrix = conf_matrix / total * 100
    
    # Create text annotations with counts and percentages
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"Count: {conf_matrix[i, j]}<br>({percentage_matrix[i, j]:.1f}%)",
                    showarrow=False,
                    font=dict(color='black' if percentage_matrix[i, j] < 70 else 'white')
                )
            )
    
    # Create custom hover text
    hover_text = [
        [f"""
        True Negative: {conf_matrix[0, 0]} ({percentage_matrix[0, 0]:.1f}%)
        False Positive: {conf_matrix[0, 1]} ({percentage_matrix[0, 1]:.1f}%)
        False Negative: {conf_matrix[1, 0]} ({percentage_matrix[1, 0]:.1f}%)
        True Positive: {conf_matrix[1, 1]} ({percentage_matrix[1, 1]:.1f}%)
        """] * 2] * 2

    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted No Failure', 'Predicted Failure'],
        y=['Actual No Failure', 'Actual Failure'],
        colorscale=[[0, '#EFF3FF'], [0.5, '#3182BD'], [1, '#08519C']],
        hoverongaps=False,
        hoverinfo='text',
        text=hover_text
    ))
    
    # Add annotations
    fig.update_layout(
        annotations=annotations,
        title={
            'text': f'Confusion Matrix - {model_name}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        xaxis={'side': 'bottom'},
        width=700,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    
    # Add a color bar title
    fig.update_traces(colorbar_title="Count")
    
    return fig

def plot_feature_importance(model, feature_names, X, y):
    """
    Plot feature importance using either built-in feature importances or permutation importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X: Feature matrix used for permutation importance
        y: Target variable used for permutation importance
    """
    importances = None
    importance_type = None
    
    # Try to get built-in feature importances (works for Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_type = 'Built-in Feature Importance'
    
    # For models without built-in feature importance, use permutation importance
    else:
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
        importance_type = 'Permutation Importance'
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title=f'{importance_type} Analysis',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=max(400, len(feature_names) * 30)  # Dynamic height based on number of features
    )
    
    return fig

def plot_feature_distributions(df):
    # Select numerical columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Target', 'Failure Type', 'UDI', 'Product ID']]
    
    # Create subplots
    fig = go.Figure()
    
    for col in numeric_cols:
        # Add histogram for each feature
        fig.add_trace(go.Histogram(
            x=df[col],
            name=col,
            visible=False,
            nbinsx=30
        ))
    
    # Make first trace visible
    fig.data[0].visible = True
    
    # Create buttons for each feature
    buttons = []
    for i, col in enumerate(numeric_cols):
        buttons.append({
            'label': col,
            'method': 'update',
            'args': [{'visible': [j == i for j in range(len(numeric_cols))]},
                    {'title': f'Distribution of {col}'}]
        })
    
    # Update layout
    fig.update_layout(
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15
        }],
        title=f'Distribution of {numeric_cols[0]}',
        height=500
    )
    
    return fig

def plot_correlation_matrix(df):
    # Calculate correlation matrix
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Target', 'Failure Type', 'UDI', 'Product ID']]
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={'size': 10}
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=700,
        width=700
    )
    
    return fig

def plot_feature_vs_target(df, feature):
    fig = go.Figure()
    
    # Add box plots for each target class
    for target in [0, 1]:
        fig.add_trace(go.Box(
            y=df[df['Target'] == target][feature],
            name=f'{"Failure" if target == 1 else "No Failure"}',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title=f'{feature} Distribution by Machine Status',
        yaxis_title=feature,
        showlegend=True,
        height=500
    )
    
    return fig

def plot_machine_type_distribution(df):
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

def plot_cv_scores(cv_results):
    # Create DataFrame for CV scores
    df = pd.DataFrame({
        'Model': [model for model in cv_results.keys()],
        'Mean CV Score': [cv_results[model]['mean_score'] for model in cv_results.keys()],
        'Std CV Score': [cv_results[model]['std_score'] for model in cv_results.keys()]
    })
    
    fig = go.Figure()
    
    # Add CV scores with error bars
    fig.add_trace(go.Bar(
        x=df['Model'],
        y=df['Mean CV Score'],
        error_y=dict(
            type='data',
            array=df['Std CV Score'],
            visible=True
        ),
        name='Cross-Validation Score'
    ))
    
    fig.update_layout(
        title='Cross-Validation Scores by Model',
        xaxis_title='Model',
        yaxis_title='CV Score',
        yaxis_range=[0, 1]
    )
    
    return fig