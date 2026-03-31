import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt

# 1. Load the massive synthetic dataset
df = pd.read_csv("orbital_features.csv")

# 2. Define Features (X) and Target (Y)
X = df[["semi_major_axis_km", "bstar"]] 
Y = df["days_to_decay"]

# 3. Train/Test Split (80% for training, 20% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows...")

# 4. Initialize and Train the Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, Y_train)

# 5. Evaluate on Unseen Data (The True Test)
predictions = rf_model.predict(X_test)

# Calculate Error Metrics
mae = mean_absolute_error(Y_test, predictions)
print("\n--- Model Performance ---")
print(f"Mean Absolute Error: {mae:.2f} days")
print("(This means our model's countdown is usually accurate within this many days)")

results_df = pd.DataFrame({
    "Actual_Days_Left": Y_test.values,
    "Predicted_Days": predictions,
    # Let's calculate the exact error for each row
    "Error_Margin": abs(Y_test.values - predictions) 
})

print("\n--- A Peek at the Actual Predictions (First 10) ---")
# We use .head(10) so it only prints the first 10 rows instead of 200
print(results_df.head(10).round(2).to_string(index=False))

print("\n--- Generating SHAP Explanations ---")

# 1. Initialize the SHAP Explainer
# TreeExplainer is heavily optimized for Random Forests
explainer = shap.TreeExplainer(rf_model)

# 2. Calculate SHAP values for the test dataset
shap_values = explainer(X_test)

# 3. GLOBAL EXPLAINABILITY (The Summary Plot)
print("Displaying Global Summary Plot (Close the plot window to continue)...")
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Global Feature Impact on Time-to-Decay")
plt.tight_layout()
plt.show()

# 4. LOCAL EXPLAINABILITY (The Waterfall Plot)
# Let's interrogate the very first prediction in our test set
row_index = 0
actual_val = Y_test.iloc[row_index]
predicted_val = predictions[row_index]

print(f"\nExplaining Prediction for Test Row {row_index}:")
print(f"Actual Days Left: {actual_val}")
print(f"Predicted Days: {predicted_val:.2f}")

print("Displaying Local Waterfall Plot...")
plt.figure(figsize=(8, 5))
# The waterfall plot shows exactly how we got from the model's average to this specific prediction
shap.plots.waterfall(shap_values[row_index], show=False)
plt.title(f"Local Explanation for Row {row_index}")
plt.tight_layout()
plt.show()