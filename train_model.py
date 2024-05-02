import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
# Read the data from the CSV file
df = pd.read_csv('dataset.csv', delimiter=";")

# Define features and target
features = df.drop('quality', axis=1)
target = df['quality']

# Separate data for training and testing (optional for this example)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features (assuming it improves the model)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define and evaluate Linear Regression model
model_lr = LinearRegression()
scores_lr = cross_val_score(model_lr, features_scaled, target, cv=5)  # 5-fold cross-validation
print("Linear Regression Cross-Validation Scores:", scores_lr)
print("Mean Cross-Validation Score (Linear Regression):", scores_lr.mean())

# Define and evaluate Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust n_estimators as needed

# Fit the Random Forest model on the scaled features and target
model_rf.fit(features_scaled, target)  # This line fixes the error

scores_rf = cross_val_score(model_rf, features_scaled, target, cv=5)
print("Random Forest Cross-Validation Scores:", scores_rf)
print("Mean Cross-Validation Score (Random Forest):", scores_rf.mean())

# Choose the model to use for prediction based on CV scores (or other criteria)
# For this example, let's use the model with the higher mean CV score (replace with your choice)
chosen_model = model_rf  # Modify this line to choose the desired model

# Save the chosen model and scaler (if used)
pickle.dump(chosen_model, open('wine_quality_model.pkl', 'wb'))
if hasattr(scaler, 'transform'):  # Save scaler if it exists
    pickle.dump(scaler, open('wine_quality_scaler.pkl', 'wb'))

# Sample input data
sample_input = "8.1;0.28;0.4;6.9;0.05;30;97;0.9951;3.26;0.44;10.1"

# Split the data into a list of floats
sample_list = [float(x) for x in sample_input.split(";")]

# Reshape the list into a 2D array
sample_input_array = np.array([sample_list])

# Apply scaling if the chosen model requires it
if hasattr(scaler, 'transform'):  # Apply scaler if it exists
    sample_input_array = scaler.transform(sample_input_array)

# Make a prediction using the chosen model (after fitting)
predicted_quality = chosen_model.predict(sample_input_array)[0]

# Print the predicted quality
print("Predicted quality for the sample:", predicted_quality)
