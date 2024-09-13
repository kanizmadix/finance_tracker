import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('Modified_Cleaned_Data.csv')
        return data
    except FileNotFoundError:
        st.error("Data file 'Modified_Cleaned_Data.csv' not found. Please make sure it's in the same directory as this script.")
        st.stop()

data = load_data()

# Preprocess the data
le_dict = {}
encoded_data = data.copy()
for column in encoded_data.columns:
    if encoded_data[column].dtype == 'object':
        le = LabelEncoder()
        encoded_data[column] = le.fit_transform(encoded_data[column].astype(str))
        le_dict[column] = le

# Train the model
X = encoded_data.drop(['9) How often do you save money each month?', '10) What percentage of your monthly pocket money/Income do you save ?'], axis=1)
y = encoded_data['10) What percentage of your monthly pocket money/Income do you save ?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title('Student Financial Analysis and Savings Predictor')

# Sidebar for user input
st.sidebar.header('Enter Your Information')

# Create input fields for user data
user_input = {}
for column in data.columns:
    if column not in ['9) How often do you save money each month?', '10) What percentage of your monthly pocket money/Income do you save ?']:
        if data[column].dtype == 'object':
            user_input[column] = st.sidebar.selectbox(column, options=[''] + list(data[column].unique()))
        else:
            user_input[column] = st.sidebar.number_input(column, min_value=0, value=0)

# Create a DataFrame from user input
user_data = pd.DataFrame(user_input, index=[0])

# Encode user input
user_data_encoded = user_data.copy()
for column in user_data_encoded.columns:
    if column in le_dict:
        le = le_dict[column]
        # Handle unseen categories
        user_data_encoded[column] = user_data_encoded[column].map(lambda x: np.nan if x not in le.classes_ else le.transform([x])[0])

# Check if there are any NaN values after encoding
if user_data_encoded.isnull().values.any():
    st.warning("Some of your input contains categories not seen in the training data. This may affect the prediction accuracy.")

# Replace NaN with a placeholder value (e.g., -1)
user_data_encoded = user_data_encoded.fillna(-1)

# Make prediction
prediction = model.predict(user_data_encoded)[0]
savings_categories = ['Less than 10%', '10-30%', 'Over 30%']
predicted_savings = savings_categories[prediction]

# Display prediction and recommendations
st.header('Your Savings Prediction')
st.write(f"Based on your input, your predicted savings category is: **{predicted_savings}**")

st.header('Recommendations')
if prediction == 0:
    st.write("To increase your savings:")
    st.write("1. Create a detailed budget and track your expenses")
    st.write("2. Cut down on non-essential expenses, especially in entertainment and dining out")
    st.write("3. Look for ways to increase your income, such as part-time jobs or freelancing")
    st.write("4. Set specific savings goals and automate your savings")
elif prediction == 1:
    st.write("You're doing well with savings, but there's room for improvement:")
    st.write("1. Try to increase your savings by an additional 5%")
    st.write("2. Look for better investment options to grow your savings")
    st.write("3. Analyze your expenses and see if you can further reduce costs in any category")
    st.write("4. Consider setting up automatic transfers to your savings account")
else:
    st.write("Great job on savings! To maintain and improve:")
    st.write("1. Diversify your investments to manage risk")
    st.write("2. Consider long-term financial goals, such as retirement planning")
    st.write("3. Regularly review and adjust your financial plan")
    st.write("4. Share your saving strategies with peers to help them improve their financial health")

# Display expense analysis
st.header('Your Expense Analysis')
expense_columns = ['b) Food and Groceries', 'c) Transportation', 'd) Utilities (Electricity, Water, Internet, etc.)', 'e) Entertainment (Movies, Concerts, etc.)', 'f) Health and Fitness (Gym, Sports, etc.)']
user_expenses = user_data[expense_columns].iloc[0]
fig, ax = plt.subplots()
user_expenses.plot(kind='bar', ax=ax)
plt.title('Your Expenses by Category')
plt.ylabel('Expense (in Rs)')
plt.xlabel('Expense Category')
plt.tight_layout()
st.pyplot(fig)

# Suggestions for reducing expenses
st.header('Suggestions for Reducing Expenses')
highest_expense = user_expenses.idxmax()
st.write(f"Your highest expense category is **{highest_expense}**. Here are some tips to reduce expenses in this category:")

if 'Food' in highest_expense:
    st.write("1. Plan your meals and grocery shopping in advance")
    st.write("2. Cook at home more often and pack lunches")
    st.write("3. Use coupons and look for discounts on groceries")
    st.write("4. Buy non-perishable items in bulk when on sale")
elif 'Transportation' in highest_expense:
    st.write("1. Use public transportation or carpooling when possible")
    st.write("2. Walk or bike for short distances")
    st.write("3. Compare fuel prices and use apps to find the cheapest gas stations")
    st.write("4. Properly maintain your vehicle to improve fuel efficiency")
elif 'Utilities' in highest_expense:
    st.write("1. Use energy-efficient appliances and light bulbs")
    st.write("2. Unplug electronics when not in use")
    st.write("3. Adjust your thermostat to save on heating/cooling costs")
    st.write("4. Compare internet and phone plans for better deals")
elif 'Entertainment' in highest_expense:
    st.write("1. Look for free or low-cost entertainment options in your area")
    st.write("2. Use student discounts for movies, concerts, and events")
    st.write("3. Host potluck gatherings instead of eating out")
    st.write("4. Consider sharing streaming subscriptions with friends or family")
elif 'Health' in highest_expense:
    st.write("1. Look for student discounts at gyms or fitness centers")
    st.write("2. Explore free workout videos or apps")
    st.write("3. Join university sports clubs or intramural teams")
    st.write("4. Prepare healthy meals at home instead of buying expensive health foods")

st.header('Next Steps')
st.write("1. Review your expenses regularly and adjust your budget as needed")
st.write("2. Set specific, achievable financial goals")
st.write("3. Educate yourself about personal finance and investing")
st.write("4. Seek advice from financial advisors or mentors if needed")

# Add a note about the model and data
st.sidebar.markdown("---")
st.sidebar.write("Note: This model is based on sample data and should be used for educational purposes only. Always consult with a financial advisor for personalized advice.")