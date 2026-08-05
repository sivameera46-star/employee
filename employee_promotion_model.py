import pandas as pd
import joblib

# Load the trained model
model = joblib.load("employee_promotion_model.pkl")

# Load test data
data_df = pd.read_csv("employee_test_data.csv")

# Prediction
y_pred = model.predict(data_df)

# Display predictions
print("Predictions:", y_pred)

# Display result for each employee
for i, pred in enumerate(y_pred, start=1):
    if pred == 1:
        print(f"Employee {i}: Promoted")
    else:
        print(f"Employee {i}: Not Promoted")