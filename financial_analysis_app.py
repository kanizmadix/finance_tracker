import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page config for wider layout
st.set_page_config(layout="wide")

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to toggle theme
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Custom CSS for dark mode styling
dark_mode_css = """
<style>
    body {
        color: white;
        background-color: black;
    }
    .stApp {
        background-color: black;
    }
    h1, h2, h3 {
        color: gold !important;
    }
    .stRadio > div {
        color: white;
    }
    .stRadio > div > label > div[role="radiogroup"] > label > div:first-child {
        background-color: black;
        border-color: white;
    }
    .stRadio > div > label > div[role="radiogroup"] > label > div:first-child::before {
        background-color: red;
    }
    .stButton>button {
        background-color: gold;
        color: black;
    }
    .highlight {
        color: gold;
        font-weight: bold;
    }
    .disclaimer {
        color: red;
        font-style: italic;
    }
    .sidebar .sidebar-content {
        background-color: black;
    }
</style>
"""

# Apply custom CSS only in dark mode
if st.session_state.dark_mode:
    st.markdown(dark_mode_css, unsafe_allow_html=True)

# Rest of your functions remain the same
def range_to_value(range_str):
    if range_str == 'No rent':
        return 0
    if 'Under' in range_str or 'Below' in range_str:
        return float(range_str.split()[-1]) / 2
    elif '-' in range_str:
        lower, upper = map(float, range_str.replace('Rs', '').split('-'))
        return (lower + upper) / 2
    elif 'Over' in range_str:
        return float(range_str.split()[-1]) * 1.25
    return 0

def calculate_financial_literacy(savings_rate):
    if savings_rate > 30:
        return "Excellent"
    elif savings_rate > 20:
        return "Good"
    elif savings_rate > 10:
        return "Fair"
    else:
        return "Poor"

st.title('💰 Student Financial Analysis and Savings Predictor')

# Theme toggle
st.sidebar.button("Toggle Dark Mode", on_click=toggle_theme)

# Move disclaimer to the left
st.sidebar.markdown('<p class="disclaimer">Note: This analysis is based on estimates and should be used for educational purposes only. Always consult with a financial advisor for personalized advice.</p>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns(2)

with col1:
    income_ranges = ['Below 3000', '3000-6000', '6000-10000', 'Over 10000']
    expense_ranges = ['Below 3000', '3000-6000', '6000-10000', 'Over 10000']
    rent_ranges = ['No rent', 'Under 3000', '3000-5000', '5000-7000', 'Over 7000']
    other_expense_ranges = ['Under 1000', '1000-2000', '2000-3000', 'Over 3000']
    food_ranges = ['Under 3000', '3000-5000', '5000-7000', 'Over 7000']
    saving_frequency = ['Always', 'Sometimes', 'Rarely', 'Never']
    saving_percentage = ['None', 'Less than 10%', '10-30%', 'Over 30%']

    income = st.radio("Monthly Pocket Money/Income (in Rs)", income_ranges)
    total_expenses = st.radio("How much do you usually spend per month (in Rs)", expense_ranges)

    st.subheader("Estimate your monthly spending in the following categories (in Rs)")
    rent = st.radio("Rent/Accommodation", rent_ranges)
    food = st.radio("Food and Groceries", food_ranges)
    transportation = st.radio("Transportation", other_expense_ranges)
    utilities = st.radio("Utilities (Electricity, Water, Internet, etc.)", other_expense_ranges)
    entertainment = st.radio("Entertainment (Movies, Concerts, etc.)", other_expense_ranges)
    health = st.radio("Health and Fitness (Gym, Sports, etc.)", other_expense_ranges)

    saving_freq = st.radio("How often do you save money each month?", saving_frequency)
    saving_percent = st.radio("What percentage of your monthly pocket money/Income do you save?", saving_percentage)

with col2:
    if st.button('Analyze My Finances'):
        income_value = range_to_value(income)
        expense_value = range_to_value(total_expenses)
        rent_value = range_to_value(rent)
        food_value = range_to_value(food)
        transportation_value = range_to_value(transportation)
        utilities_value = range_to_value(utilities)
        entertainment_value = range_to_value(entertainment)
        health_value = range_to_value(health)

        total_expenses_calculated = sum([rent_value, food_value, transportation_value, utilities_value, entertainment_value, health_value])
        savings = income_value - total_expenses_calculated

        if saving_percent == 'None':
            savings_rate = 0
        elif saving_percent == 'Less than 10%':
            savings_rate = 5
        elif saving_percent == '10-30%':
            savings_rate = 20
        else:
            savings_rate = 35

        financial_literacy = calculate_financial_literacy(savings_rate)

        st.subheader('Your Financial Summary')
        st.write(f"Total Monthly Income: Rs. {income_value:.2f}")
        st.write(f"Total Monthly Expenses: Rs. {total_expenses_calculated:.2f}")
        st.write(f"Monthly Savings: Rs. {savings:.2f}")
        st.write(f"Savings Rate: {savings_rate}%")
        st.write(f"Financial Literacy: {financial_literacy}")

        expense_categories = ['Rent', 'Food', 'Transportation', 'Utilities', 'Entertainment', 'Health']
        expense_values = [rent_value, food_value, transportation_value, utilities_value, entertainment_value, health_value]

        if rent == 'No rent':
            expense_categories = expense_categories[1:]
            expense_values = expense_values[1:]

        fig, ax = plt.subplots(facecolor='black' if st.session_state.dark_mode else 'white')
        colors = plt.cm.Set3(np.linspace(0, 1, len(expense_categories)))
        wedges, texts, autotexts = ax.pie(expense_values, labels=expense_categories, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        plt.setp(texts, color="white" if st.session_state.dark_mode else "black")
        plt.setp(autotexts, color="black" if st.session_state.dark_mode else "white", weight="bold")
        plt.title('Your Expense Distribution', color='white' if st.session_state.dark_mode else 'black')
        st.pyplot(fig)

        st.subheader('Recommendations')
        if savings_rate < 10:
            st.write("Your savings rate is low. Consider the following:")
            st.markdown("1. Create a detailed budget to track your expenses")
            st.markdown("2. Look for areas to cut back on non-essential spending")
            st.markdown("3. Try to save at least 10% of your income")
            st.markdown("4. Consider finding additional sources of income")
        elif savings_rate < 20:
            st.write("You're on the right track, but there's room for improvement:")
            st.markdown("1. Aim to increase your savings to 20% of your income")
            st.markdown("2. Review your expenses and see where you can cut back")
            st.markdown("3. Consider automating your savings")
            st.markdown("4. Look into low-risk investment options")
        else:
            st.write("Great job on your savings! Here are some tips to maintain and improve:")
            st.markdown("1. Keep up your excellent saving habits")
            st.markdown("2. Consider diversifying your investments")
            st.markdown("3. Set long-term financial goals")
            st.markdown("4. Share your financial knowledge with others")

        st.markdown("### Investment Suggestions")
        st.markdown("Consider investing your savings in:")
        st.markdown('<p class="highlight">• Stocks: Potential for high returns, but comes with higher risk</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">• Mutual Funds: Professionally managed, diversified investment portfolios</p>', unsafe_allow_html=True)
        st.markdown('<p class="highlight">• SIPs (Systematic Investment Plans): Regular, disciplined investing in mutual funds</p>', unsafe_allow_html=True)
        st.markdown('<p class="disclaimer">Disclaimer: Before investing, ensure you have proper knowledge of how the market works. Investments carry risks, and it\'s important to understand these risks before making any investment decisions.</p>', unsafe_allow_html=True)

        st.subheader('The Importance of Saving')
        st.write("Saving money is crucial for your financial future. It provides a safety net for emergencies, helps you achieve your goals, and sets you up for long-term financial success. Even small amounts saved regularly can make a big difference over time!")