import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv("faizan2.csv")

# Split inputs and outputs
X = data.iloc[:, :11]  # Inputs (chemicals)
y = data.iloc[:, 11:]  # Outputs (flyash)

# Train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Page title and layout
st.set_page_config(page_title="The Ash Oracle", page_icon=":chart_with_upwards_trend:", layout="wide")

# Header
st.title("The Ash Oracle")
st.write("Predict the flyash outputs based on chemical inputs.")

# Sidebar
st.sidebar.header("Input Chemicals")
input_values = []
for i in range(11):
    input_values.append(st.sidebar.number_input(f"Chemical {i+1}", value=0.0, step=0.1))

# Button to calculate
if st.sidebar.button("Predict", key="calculate_button"):
    # Validate input values
    if all(input_values):
        # Predict flyash outputs
        flyash_outputs = model.predict([input_values])[0]
        
        # Calculate sum of predicted values
        predicted_sum = sum(flyash_outputs)
        
        # Assign class label based on sum
        if predicted_sum > 70:
            flyash_class = "Class F"
        elif predicted_sum >= 50:
            flyash_class = "Class C"
        else:
            flyash_class = "Unclassified"
        
        # Display results
        st.header("Predicted Flyash Outputs")
        for i, output in enumerate(flyash_outputs):
            st.write(f"Flyash Output {i+1}: {output}")
        
        st.write(f"Sum of Predicted Values: {predicted_sum}")
        st.write(f"Class Label: {flyash_class}")
    else:
        st.error("Please provide values for all input fields.")
