
# Importing necessary libraries for data manipulation and regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, max_error

# Step 1: Define the dataset based on the problem statement
data = {
    "Date": [
        "2024-01-01", "2024-01-02", "2024-01-03",
        "2024-01-04", "2024-01-05", "2024-01-06"
    ],
    "Product": ["Widget", "Widget", "Gadget", "Widget", "Gadget", "Widget"],
    "Amount": [100, 150, 200, 120, 180, 130],
    "Quantity": [2, 3, 5, 4, 2, 3]
}

# Step 2: Create a DataFrame from the dataset
df = pd.DataFrame(data)

# Step 3: Add a new column for sales by multiplying Amount and Quantity
df["Sales"] = df["Amount"] * df["Quantity"]

# Step 4: Prepare the data for regression analysis
# 'Amount' and 'Quantity' are used as features (independent variables)
# 'Sales' is the target (dependent variable)
X = df[["Amount", "Quantity"]]
y = df["Sales"]

# Step 5: Split the dataset into training and testing sets (67% train, 33% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 6: Initialize the Linear Regression model
model = LinearRegression()

# Step 7: Train the model on the training data
model.fit(X_train, y_train)

# Step 8: Make predictions on the testing data
y_pred = model.predict(X_test)

# Step 9: Evaluate the model using regression metrics
scores = {
    "Mean Squared Error": mean_squared_error(y_test, y_pred),
    "R2 Score": r2_score(y_test, y_pred),
    "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
    "Explained Variance Score": explained_variance_score(y_test, y_pred),
    "Max Error": max_error(y_test, y_pred)
}

# Step 10: Display the dataset and regression scores
print("Dataset:\n", df)
print("\nRegression Scores:\n", scores)
