# This script will install required libraries and perform financial analysis

# First, install required libraries
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']

# Install packages
for package in packages:
    install(package)

print("All required packages have been installed.")

# Now, import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
# Use raw string (r prefix) for the file path
data = pd.read_csv(r'C:\Users\Kaniz\Pictures\V Project\Modified_Cleaned_Data.csv')

# Preprocess the data
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column].astype(str))

# 1. Expense Analyzer
def expense_analyzer(data):
    expense_columns = ['b) Food and Groceries', 'c) Transportation', 'd) Utilities (Electricity, Water, Internet, etc.)', 'e) Entertainment (Movies, Concerts, etc.)', 'f) Health and Fitness (Gym, Sports, etc.)']
    
    expense_data = data[expense_columns]
    expense_mean = expense_data.mean()
    
    plt.figure(figsize=(10, 6))
    expense_mean.plot(kind='bar')
    plt.title('Average Expenses by Category')
    plt.ylabel('Average Expense (Encoded Value)')
    plt.xlabel('Expense Category')
    plt.tight_layout()
    plt.savefig('expense_analysis.png')
    plt.close()
    
    return expense_mean

# 2. Gender-wise Financial Literacy Analysis
def gender_financial_literacy(data):
    gender_literacy = data.groupby('1) Gender')['13) How would you rate your financial literacy?'].mean()
    
    plt.figure(figsize=(8, 5))
    gender_literacy.plot(kind='bar')
    plt.title('Average Financial Literacy by Gender')
    plt.ylabel('Average Financial Literacy (Encoded Value)')
    plt.xlabel('Gender')
    plt.tight_layout()
    plt.savefig('gender_literacy_analysis.png')
    plt.close()
    
    return gender_literacy

# 3. Predictive Model for Savings
def savings_predictor(data):
    X = data.drop(['9) How often do you save money each month?', '10) What percentage of your monthly pocket money/Income do you save ?'], axis=1)
    y = data['10) What percentage of your monthly pocket money/Income do you save ?']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 10 Features for Predicting Savings')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return model

# Run the analyses
expense_analysis = expense_analyzer(data)
gender_literacy_analysis = gender_financial_literacy(data)
savings_model = savings_predictor(data)

print("Expense Analysis:")
print(expense_analysis)
print("\nGender-wise Financial Literacy Analysis:")
print(gender_literacy_analysis)

# Function to predict savings and provide recommendations
def predict_and_recommend(model, user_data):
    user_data_encoded = user_data.copy()
    for column in user_data_encoded.columns:
        if user_data_encoded[column].dtype == 'object':
            user_data_encoded[column] = le.transform(user_data_encoded[column].astype(str))
    
    prediction = model.predict(user_data_encoded)[0]
    
    savings_categories = ['Less than 10%', '10-30%', 'Over 30%']
    predicted_savings = savings_categories[prediction]
    
    print(f"Predicted savings: {predicted_savings}")
    
    if prediction == 0:
        print("Recommendations to increase savings:")
        print("1. Create a budget and track your expenses")
        print("2. Cut down on non-essential expenses")
        print("3. Look for ways to increase your income")
        print("4. Set specific savings goals")
    elif prediction == 1:
        print("You're doing well with savings, but there's room for improvement:")
        print("1. Try to increase your savings by 5%")
        print("2. Look for better investment options")
        print("3. Automate your savings")
    else:
        print("Great job on savings! To maintain and improve:")
        print("1. Diversify your investments")
        print("2. Consider long-term financial goals")
        print("3. Regularly review and adjust your financial plan")

# Example usage of the predict_and_recommend function
# Note: You would need to provide actual user data in the same format as the original dataset
# user_data = pd.DataFrame({...})  # Fill this with user input
# predict_and_recommend(savings_model, user_data)

print("Analysis complete. Check the generated PNG files for visualizations.")