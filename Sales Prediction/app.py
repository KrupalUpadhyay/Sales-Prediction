# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from io import BytesIO
import base64

app = Flask(__name__)
model = joblib.load('sales_prediction_model.pkl')
data = pd.read_csv('assets/Advertising.csv')

# Function for feature engineering
def perform_feature_engineering(data):
    # Interaction terms and polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    interactions = poly.fit_transform(data[['TV', 'Radio', 'Newspaper']])
    interactions_df = pd.DataFrame(interactions, columns=poly.get_feature_names_out(['TV', 'Radio', 'Newspaper']))
    data = pd.concat([data, interactions_df], axis=1)
    return data

# Function for model comparison
def compare_models(X, y):
    models = [
        ('Linear Regression', Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    results = []
    for name, model in models:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        results.append((name, cv_scores.mean(), cv_scores.std()))
    return results

# Perform feature engineering on data
data = perform_feature_engineering(data)

# Split data into X and y
X = data.drop('Sales', axis=1)
y = data['Sales']

# Compare models
model_comparison_results = compare_models(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            tv = float(request.form['tv'])
            radio = float(request.form['radio'])
            newspaper = float(request.form['newspaper'])
            
            # Create input data as DataFrame
            input_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Generate feature importance plot
            feature_importances = model.feature_importances_
            features = input_data.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.xticks(rotation=45)
            
            # Save plot to a bytes object
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            # Render results template with prediction, plot, and model comparison results
            return render_template('results.html', prediction=prediction[0], plot_url=plot_url, model_results=model_comparison_results)
        
        except ValueError:
            return render_template('index.html', error_message="Please enter valid numerical values.")

if __name__ == '__main__':
    app.run(debug=True)
