import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app
st.title('Student Financial Analysis and Savings Calculator')

# User input
income_source = st.selectbox('Primary Source of Income', ['Parental Support', 'Part Time Job', 'Others (specify)'])

income_ranges = ['500-1000', '1000-1500', '1500-2000', 'Over 2000']
monthly_income = st.selectbox('Monthly Pocket Money/Income(in Rs)', income_ranges)

expense_tracking = st.selectbox('How do you track your expenses?', 
                                ['I don\'t track my expenses', 'Mobile App', 'Spreadsheet', 'Paper and Pen'])

spending_ranges = ['Under 500', '500-800', '800-1500', 'Over 1500']
monthly_spending = st.selectbox('How much do you usually spend per month(in Rs)', spending_ranges)

# Expense categories
expense_categories = [
    'a) Rent/Accommodation',
    'b) Food and Groceries',
    'c) Transportation',
    'd) Utilities (Electricity, Water, Internet, etc.)',
    'e) Entertainment (Movies, Concerts, etc.)',
    'f) Health and Fitness (Gym, Sports, etc.)'
]

st.subheader('Estimate your monthly spending in the following categories (in Rs):')
expenses = {}
for category in expense_categories:
    expenses[category] = st.selectbox(category, ['Under 500', '500-1000', '1000-1500', 'Over 1500'])

# Convert range to numeric value for calculation
def range_to_value(range_str):
    if range_str == 'Under 500':
        return 250
    elif range_str == '500-800':
        return 650
    elif range_str == '500-1000':
        return 750
    elif range_str == '800-1500':
        return 1150
    elif range_str == '1000-1500':
        return 1250
    elif range_str == '1500-2000':
        return 1750
    elif range_str == 'Over 1500' or range_str == 'Over 2000':
        return 2000
    else:
        return 0

# Calculate total expenses and create pie chart
total_expenses = sum(range_to_value(value) for value in expenses.values())

fig, ax = plt.subplots()
plt.pie([range_to_value(value) for value in expenses.values()], 
        labels=[cat.split(') ')[1] for cat in expense_categories], 
        autopct='%1.1f%%')
plt.title('Your Expense Distribution')
st.pyplot(fig)

# Calculate and display savings rate
income_value = range_to_value(monthly_income)
savings = income_value - total_expenses
savings_rate = (savings / income_value) * 100 if income_value > 0 else 0

st.subheader('Your Financial Summary')
st.write(f"Total Monthly Income: Rs. {income_value}")
st.write(f"Total Monthly Expenses: Rs. {total_expenses}")
st.write(f"Monthly Savings: Rs. {savings}")
st.write(f"Savings Rate: {savings_rate:.2f}%")

# Provide recommendations
st.subheader('Recommendations')
if savings_rate < 10:
    st.write("Your savings rate is low. Here are some recommendations:")
    st.write("1. Create a detailed budget to track your expenses")
    st.write("2. Look for areas to cut back, especially in your highest expense categories")
    st.write("3. Consider finding additional sources of income")
    st.write("4. Aim to save at least 10% of your income")
elif savings_rate < 20:
    st.write("You're saving, but there's room for improvement:")
    st.write("1. Try to increase your savings rate to 20%")
    st.write("2. Review your expenses and see if you can reduce any further")
    st.write("3. Consider setting up automatic transfers to a savings account")
    st.write("4. Look into low-risk investment options for your savings")
else:
    st.write("Great job on saving! Here are some tips to maintain and improve:")
    st.write("1. Continue your good saving habits")
    st.write("2. Consider diversifying your savings into different investment options")
    st.write("3. Set long-term financial goals")
    st.write("4. Educate yourself on personal finance and investing")

# Specific recommendations based on highest expense category
highest_expense = max(expenses, key=lambda k: range_to_value(expenses[k]))
st.subheader('Tips for Reducing Your Highest Expense')
st.write(f"Your highest expense category is {highest_expense.split(') ')[1]}. Here are some tips to reduce it:")

if 'Rent' in highest_expense:
    st.write("1. Consider shared accommodation or a less expensive area")
    st.write("2. Negotiate with your landlord for a better rate")
    st.write("3. Look for inclusive deals where utilities are covered")
elif 'Food' in highest_expense:
    st.write("1. Cook meals at home and pack lunches")
    st.write("2. Buy groceries in bulk and look for deals")
    st.write("3. Limit eating out and use student discounts when you do")
elif 'Transportation' in highest_expense:
    st.write("1. Use public transportation or carpooling")
    st.write("2. Walk or bike for short distances")
    st.write("3. Look for student discounts on transportation passes")
elif 'Utilities' in highest_expense:
    st.write("1. Be mindful of energy usage, turn off lights and unplug devices")
    st.write("2. Use energy-efficient appliances")
    st.write("3. Compare providers for better rates on internet and phone plans")
elif 'Entertainment' in highest_expense:
    st.write("1. Look for free or low-cost entertainment options")
    st.write("2. Use student discounts for movies and events")
    st.write("3. Consider sharing subscription services with friends")
elif 'Health' in highest_expense:
    st.write("1. Look for student discounts at gyms or fitness centers")
    st.write("2. Utilize free workout resources online or on campus")
    st.write("3. Prepare healthy meals at home instead of buying expensive health foods")

st.sidebar.info("Note: This tool is for educational purposes only. Please consult a financial advisor for personalized advice.")