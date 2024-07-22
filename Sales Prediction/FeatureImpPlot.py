import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your data
data = pd.read_csv('assets/Advertising.csv')  # replace 'your_dataset.csv' with your actual file path

# Define the features (X) and target (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Get the feature importance (coefficients)
coefficients = linear_model.coef_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(features, coefficients, color='blue', edgecolor='k', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Feature Importance')
plt.show()
