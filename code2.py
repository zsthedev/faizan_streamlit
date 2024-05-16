import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = pd.read_csv("faizan2.csv")

# Step 2: Split inputs and outputs
X = data.iloc[:, :11]  # Inputs (chemicals)
y = data.iloc[:, 11:]  # Outputs (flyash)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_rmse = mean_squared_error(y_train, train_pred, squared=False)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R^2:", train_r2)
print("Test R^2:", test_r2)

# Step 6: Add column with "Class F" or "Class C" based on predicted values
data['Class'] = 'Class C'
data.loc[data.iloc[:, 13] > 70, 'Class'] = 'Class F'

# Display the updated DataFrame
print(data)
