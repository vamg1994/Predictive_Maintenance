import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Drop UDI and Product ID columns as they're not relevant for prediction
    df = df.drop(['UDI', 'Product ID'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])
    df['Failure Type'] = le.fit_transform(df['Failure Type'])
    
    # Verify all features are numerical after encoding
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) != len(df.columns):
        raise ValueError("Some columns are not numeric after encoding")
    
    # Split features and target
    X = df.drop(['Target', 'Failure Type'], axis=1)
    y = df['Target']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def prepare_single_prediction(data, scaler):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Scale the features
    scaled_features = scaler.transform(df)
    
    return scaled_features
