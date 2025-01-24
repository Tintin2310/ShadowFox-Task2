import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('car.csv')

# Check the first few rows of the dataset to verify the format
print(df.head())

# Features and target variable
X = df.drop(columns=['Selling_Price', 'Car_Name'])  # Exclude 'Car_Name' and 'Selling_Price' as it's the target
y = df['Selling_Price']

# Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline with OneHotEncoding for categorical variables and scaling numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('fuel_type', OneHotEncoder(), ['Fuel_Type']),
        ('seller_type', OneHotEncoder(), ['Seller_Type']),
        ('transmission', OneHotEncoder(), ['Transmission']),
        ('scaler', StandardScaler(), ['Year', 'Present_Price', 'Kms_Driven', 'Owner'])
    ])

# Create the pipeline with GradientBoostingRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predictions and evaluation
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model to a file
with open('car_price_model_gbr.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model training complete and saved to 'car_price_model_gbr.pkl'")
