import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load your dataset (Excel file)
file_path = '/Users/reinamercy/wait-time-prediction/dseb/mock_up_patient_data.xlsx'
data = pd.read_excel(file_path)

# Select relevant columns (update 'post-consultation_time' if needed)
df = data[['patient_id', 'patient_age', 'patient_gender', 'financial_class', 
           'condition', 'entry_time', 'post-consultation_time']]  # Confirm this matches

# Function to convert time string to minutes since midnight
def time_to_minutes(time_str):
    if isinstance(time_str, str):
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    else:
        time_obj = datetime.strptime(time_str.strftime('%H:%M:%S'), '%H:%M:%S')
    return time_obj.hour * 60 + time_obj.minute + time_obj.second / 60

# Calculate wait time
df['entry_minutes'] = df['entry_time'].apply(time_to_minutes)
df['post_consultation_minutes'] = df['post-consultation_time'].apply(time_to_minutes)
df['wait_time'] = df['post_consultation_minutes'] - df['entry_minutes']

# Handle negative wait times and outliers
df.loc[df['wait_time'] < 0, 'wait_time'] = np.nan
df = df.dropna(subset=['wait_time'])
df = df[df['wait_time'] <= 60]  # Cap at 1 hour

# Feature engineering
df['entry_hour'] = df['entry_minutes'] // 60
df['queue_load'] = df.groupby('entry_hour')['patient_id'].transform('count')

# Severity scores
severity_scores = {
    'Acute Myocardial Infarction (AMI)': 5,
    'Emergency Department (ED) Measures': 3,
    'Blood clot prevention and treatment': 2,
    'Preventative Care Measures': 1
}
df['severity'] = df['condition'].map(severity_scores)

# New features
df = df.sort_values('entry_minutes')
df['queue_position'] = df.groupby('entry_hour').cumcount() + 1
df['prev_post_consultation'] = df['post_consultation_minutes'].shift(1).fillna(df['entry_minutes'].min())
df['time_since_last'] = df['entry_minutes'] - df['prev_post_consultation']
df['time_since_last'] = df['time_since_last'].clip(lower=0)

# Encode categorical variables
le_gender = LabelEncoder()
le_financial = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['patient_gender'])
df['financial_encoded'] = le_financial.fit_transform(df['financial_class'])

# Features and target (log transform wait_time)
features = ['patient_age', 'gender_encoded', 'financial_encoded', 'severity', 
            'entry_hour', 'queue_load', 'queue_position', 'time_since_last']
X = df[features]
y = np.log1p(df['wait_time'])  # Log transform

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Diagnostics
print("Sample of processed data:")
print(df[features + ['wait_time']].head())
print("\nWait Time Statistics (before log transform):")
print(df['wait_time'].describe())
print("\nLog Wait Time Statistics:")
print(pd.Series(y).describe())
print("\nCorrelation with Log Wait Time:")
print(pd.concat([X, y.rename('log_wait_time')], axis=1).corr()['log_wait_time'].sort_values())

# --- XGBoost Model with Grid Search ---
xgb_base = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [1000, 1500, 2000],
    'learning_rate': [0.001, 0.005, 0.01],
    'max_depth': [5, 7, 10]
}
grid_search = GridSearchCV(xgb_base, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
xgb_model = grid_search.best_estimator_
print(f"\nBest Parameters: {grid_search.best_params_}")

# Predict and evaluate (inverse log transform for RMSE)
y_pred_log = xgb_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test_original, y_pred))
print(f"XGBoost RMSE: {rmse_xgb:.2f} minutes")

# Feature importance
importances = xgb_model.feature_importances_
print("\nFeature Importance (XGBoost):")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

def predict_wait_time(patient_age, gender, financial_class, condition, entry_time, queue_load, queue_position, time_since_last):
    """
    Predict wait time in minutes based on input features.
    
    Args:
        patient_age (int): Age of the patient
        gender (str): 'male', 'female', or 'unidentified'
        financial_class (str): 'HMO', 'medicare', 'insurance', 'corporate'
        condition (str): One of the conditions in severity_scores
        entry_time (str): Time in 'HH:MM:SS' format
        queue_load (int): Number of patients in the queue for that hour
        queue_position (int): Patient's position in the queue
        time_since_last (float): Minutes since last patient's consultation ended
    
    Returns:
        float: Predicted wait time in minutes
    """
    # Prepare input data
    input_data = pd.DataFrame({
        'patient_age': [patient_age],
        'patient_gender': [gender],
        'financial_class': [financial_class],
        'condition': [condition],
        'entry_time': [entry_time],
        'queue_load': [queue_load],
        'queue_position': [queue_position],
        'time_since_last': [time_since_last]
    })

    # Process features
    input_data['entry_minutes'] = input_data['entry_time'].apply(time_to_minutes)
    input_data['entry_hour'] = input_data['entry_minutes'] // 60
    input_data['severity'] = input_data['condition'].map(severity_scores)
    input_data['gender_encoded'] = le_gender.transform(input_data['patient_gender'])
    input_data['financial_encoded'] = le_financial.transform(input_data['financial_class'])

    # Ensure all features are present
    X_input = input_data[features]

    # Predict (log scale) and convert back to minutes
    pred_log = xgb_model.predict(X_input)
    pred_minutes = np.expm1(pred_log)[0]

    return pred_minutes

# Example usage
example_wait_time = predict_wait_time(
    patient_age=45,
    gender='male',
    financial_class='insurance',
    condition='Emergency Department (ED) Measures',
    entry_time='14:30:00',
    queue_load=5,
    queue_position=3,
    time_since_last=10.0
)
print(f"Predicted Wait Time: {example_wait_time:.2f} minutes")

wait_time = predict_wait_time(
    patient_age=72,
    gender='female',
    financial_class='medicare',
    condition='Acute Myocardial Infarction (AMI)',
    entry_time='08:45:00',
    queue_load=8,
    queue_position=5,
    time_since_last=15.0
)
print(f"Predicted Wait Time: {wait_time:.2f} minutes")

wait_time = predict_wait_time(
    patient_age=25,
    gender='male',
    financial_class='insurance',
    condition='Preventative Care Measures',
    entry_time='14:15:00',
    queue_load=3,
    queue_position=2,
    time_since_last=5.0
)
print(f"Predicted Wait Time: {wait_time:.2f} minutes")

wait_time = predict_wait_time(
    patient_age=50,
    gender='unidentified',
    financial_class='corporate',
    condition='Blood clot prevention and treatment',
    entry_time='18:30:00',
    queue_load=6,
    queue_position=4,
    time_since_last=20.0
)
print(f"Predicted Wait Time: {wait_time:.2f} minutes")
wait_time = predict_wait_time(
    patient_age=17,
    gender='female',
    financial_class='HMO',
    condition='Emergency Department (ED) Measures',
    entry_time='23:00:00',
    queue_load=2,
    queue_position=1,
    time_since_last=10.0
)
print(f"Predicted Wait Time: {wait_time:.2f} minutes")

queue = [
    {"patient_id": "P1", "age": 72, "gender": "female", "financial": "medicare", "condition": "Acute Myocardial Infarction (AMI)", "entry_time": "08:45:00", "queue_load": 8, "position": 1, "time_since_last": 15.0},
    {"patient_id": "P2", "age": 25, "gender": "male", "financial": "insurance", "condition": "Preventative Care Measures", "entry_time": "08:46:00", "queue_load": 8, "position": 2, "time_since_last": 15.0},
    {"patient_id": "P3", "age": 50, "gender": "unidentified", "financial": "corporate", "condition": "Blood clot prevention and treatment", "entry_time": "08:47:00", "queue_load": 8, "position": 3, "time_since_last": 15.0},
    {"patient_id": "P4", "age": 17, "gender": "female", "financial": "HMO", "condition": "Emergency Department (ED) Measures", "entry_time": "08:48:00", "queue_load": 8, "position": 4, "time_since_last": 15.0},
    {"patient_id": "P5", "age": 35, "gender": "male", "financial": "insurance", "condition": "Acute Myocardial Infarction (AMI)", "entry_time": "08:49:00", "queue_load": 8, "position": 5, "time_since_last": 15.0}
]

# Predict wait times for the queue
queue_df = pd.DataFrame(queue)
queue_df['predicted_wait_time'] = queue_df.apply(
    lambda row: predict_wait_time(row['age'], row['gender'], row['financial'], row['condition'], 
                                  row['entry_time'], row['queue_load'], row['position'], row['time_since_last']),
    axis=1
)

# Visualization
plt.figure(figsize=(12, 6))
bars = plt.bar(queue_df['patient_id'], queue_df['predicted_wait_time'], color='skyblue')

# Add severity as text above bars
for bar, severity in zip(bars, queue_df['condition'].map(severity_scores)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'Sev: {severity}', 
             ha='center', va='bottom', fontsize=10)

plt.xlabel('Patient ID')
plt.ylabel('Predicted Wait Time (minutes)')
plt.title('Hospital Queue Visualization')
plt.xticks(rotation=45)
plt.ylim(0, max(queue_df['predicted_wait_time']) + 10)

# Show plot
plt.tight_layout()
plt.show()

# Print queue details
print("\nQueue Details:")
print(queue_df[['patient_id', 'condition', 'entry_time', 'queue_position', 'predicted_wait_time']])