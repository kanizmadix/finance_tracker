import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Create folders for plots and output
plot_folder = 'financial_analysis_plots'
os.makedirs(plot_folder, exist_ok=True)

# Function to convert text to image
def text_to_image(text, filename):
    lines = text.split('\n')
    font_size = 14
    font = ImageFont.load_default()
    line_height = font_size + 4
    img_height = line_height * len(lines) + 20
    img_width = max(font.getlength(line) for line in lines) + 20

    image = Image.new('RGB', (int(img_width), img_height), color='white')
    draw = ImageDraw.Draw(image)
    
    y_text = 10
    for line in lines:
        draw.text((10, y_text), line, font=font, fill=(0, 0, 0))
        y_text += line_height

    image.save(filename)

# Load data
def load_data():
    data = pd.read_csv('Modified_Cleaned_Data edited.csv')
    return data

# Function to convert range to numeric value
def range_to_value(range_str):
    if isinstance(range_str, str):
        if 'Under' in range_str:
            return float(range_str.split()[-1]) / 2
        elif '-' in range_str:
            lower, upper = map(float, range_str.replace('Rs', '').split('-'))
            return (lower + upper) / 2
        elif 'Over' in range_str:
            return float(range_str.split()[-1]) * 1.25
    return 0

# 1. Income Distribution Analysis
def income_distribution_analysis(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['5) Monthly Pocket Money/Income(in Rs)'], kde=True, bins=20)
    plt.title('Distribution of Monthly Income')
    plt.xlabel('Monthly Income (Rs)')
    plt.ylabel('Count')
    plt.savefig(f'{plot_folder}/income_distribution.png')
    plt.close()
    
    print("Income Statistics:")
    print(data['5) Monthly Pocket Money/Income(in Rs)'].describe())

# 2. Spending Pattern Analysis
def spending_pattern_analysis(data):
    spending_columns = ['b) Provisions ', 'c) Transportation', 'd) Utilities (Electricity, Water, Internet, etc.)',
                        'e) Entertainment (Movies, Concerts, etc.)', 'f) Health and Fitness (Gym, Sports, etc.)']
    
    avg_spending = data[spending_columns].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    avg_spending.plot(kind='bar')
    plt.title('Average Monthly Spending by Category')
    plt.xlabel('Category')
    plt.ylabel('Average Spending (Rs)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/spending_pattern.png')
    plt.close()
    
    print("Average Spending by Category:")
    print(avg_spending)

# 3. Savings Rate Analysis
def savings_rate_analysis(data):
    data['Savings Rate'] = (data['5) Monthly Pocket Money/Income(in Rs)'] - data['7) How much do you usually spend per month(in Rs)']) / data['5) Monthly Pocket Money/Income(in Rs)'] * 100
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Savings Rate'], kde=True, bins=20)
    plt.title('Distribution of Savings Rate')
    plt.xlabel('Savings Rate (%)')
    plt.ylabel('Count')
    plt.savefig(f'{plot_folder}/savings_rate_distribution.png')
    plt.close()
    
    print("Savings Rate Statistics:")
    print(data['Savings Rate'].describe())

# 4. Financial Literacy and Savings Correlation
def financial_literacy_savings_correlation(data):
    literacy_mapping = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    data['Literacy_Numeric'] = data['13) How would you rate your financial literacy?'].map(literacy_mapping)
    
    correlation, p_value = stats.pearsonr(data['Literacy_Numeric'], data['Savings Rate'])
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='13) How would you rate your financial literacy?', y='Savings Rate', data=data, order=['Poor', 'Fair', 'Good', 'Excellent'])
    plt.title('Financial Literacy vs Savings Rate')
    plt.xlabel('Financial Literacy Level')
    plt.ylabel('Savings Rate (%)')
    plt.savefig(f'{plot_folder}/literacy_vs_savings.png')
    plt.close()
    
    print(f"Correlation between Financial Literacy and Savings Rate: {correlation:.2f}")
    print(f"P-value: {p_value:.4f}")

# 5. Expense Tracking Method Analysis
def expense_tracking_analysis(data):
    tracking_methods = data['6) How do you track your expenses?'].value_counts()
    
    plt.figure(figsize=(10, 6))
    tracking_methods.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Expense Tracking Methods')
    plt.ylabel('')
    plt.savefig(f'{plot_folder}/expense_tracking_methods.png')
    plt.close()
    
    print("Expense Tracking Methods:")
    print(tracking_methods)

# 6. Income Source Analysis
def income_source_analysis(data):
    income_sources = data['4) Primary Source of Income'].value_counts()
    
    plt.figure(figsize=(10, 6))
    income_sources.plot(kind='bar')
    plt.title('Primary Sources of Income')
    plt.xlabel('Income Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/income_sources.png')
    plt.close()
    
    print("Income Sources:")
    print(income_sources)

# 7. Gender-based Financial Behavior Analysis
def gender_financial_analysis(data):
    gender_savings = data.groupby('1) Gender')['Savings Rate'].mean()
    gender_literacy = data.groupby('1) Gender')['Literacy_Numeric'].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    gender_savings.plot(kind='bar', ax=ax1)
    ax1.set_title('Average Savings Rate by Gender')
    ax1.set_ylabel('Savings Rate (%)')
    
    gender_literacy.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Financial Literacy by Gender')
    ax2.set_ylabel('Financial Literacy Score')
    
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/gender_financial_analysis.png')
    plt.close()
    
    print("Gender-based Financial Behavior:")
    print("Average Savings Rate by Gender:")
    print(gender_savings)
    print("\nAverage Financial Literacy by Gender:")
    print(gender_literacy)

# 8. Investment Behavior Analysis
def investment_behavior_analysis(data):
    investment_rates = data['11) Do you invest your money?'].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(10, 6))
    investment_rates.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Investment Behavior')
    plt.ylabel('')
    plt.savefig(f'{plot_folder}/investment_behavior.png')
    plt.close()
    
    if '12) If yes, in which type of investment do you primarily invest ?' in data.columns:
        investment_types = data[data['11) Do you invest your money?'] == 'Yes']['12) If yes, in which type of investment do you primarily invest ?'].value_counts()
        
        plt.figure(figsize=(10, 6))
        investment_types.plot(kind='bar')
        plt.title('Primary Investment Types')
        plt.xlabel('Investment Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{plot_folder}/investment_types.png')
        plt.close()
        
        print("Investment Types:")
        print(investment_types)
    
    print("Investment Rates:")
    print(investment_rates)

# 9. Financial Challenges Analysis
def financial_challenges_analysis(data):
    challenges = data['14) What is your biggest financial challenge as a student?'].value_counts()
    
    plt.figure(figsize=(12, 6))
    challenges.plot(kind='bar')
    plt.title('Biggest Financial Challenges for Students')
    plt.xlabel('Challenge')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/financial_challenges.png')
    plt.close()
    
    print("Financial Challenges:")
    print(challenges)

# 10. Financial Confidence Analysis
def financial_confidence_analysis(data):
    confidence_levels = data['15) How confident are you in your ability to manage your finances?'].value_counts()
    
    plt.figure(figsize=(10, 6))
    confidence_levels.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Financial Confidence Levels')
    plt.ylabel('')
    plt.savefig(f'{plot_folder}/financial_confidence.png')
    plt.close()
    
    print("Financial Confidence Levels:")
    print(confidence_levels)

# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data()

    # Convert range columns to numeric
    range_columns = ['5) Monthly Pocket Money/Income(in Rs)', '7) How much do you usually spend per month(in Rs)',
                     '8) Estimate your monthly spending in the following(in Rs)\na) Rent/Accommodation', 'b) Provisions ', 'c) Transportation',
                     'd) Utilities (Electricity, Water, Internet, etc.)', 'e) Entertainment (Movies, Concerts, etc.)',
                     'f) Health and Fitness (Gym, Sports, etc.)']

    for col in range_columns:
        data[col] = data[col].apply(range_to_value)

    # Capture output and run analyses
    output_file = 'analysis_output.txt'
    with open(output_file, 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f
        
        # Run all analyses
        print("1. Income Distribution Analysis")
        income_distribution_analysis(data)

        print("\n2. Spending Pattern Analysis")
        spending_pattern_analysis(data)

        print("\n3. Savings Rate Analysis")
        savings_rate_analysis(data)

        print("\n4. Financial Literacy and Savings Correlation")
        financial_literacy_savings_correlation(data)

        print("\n5. Expense Tracking Method Analysis")
        expense_tracking_analysis(data)

        print("\n6. Income Source Analysis")
        income_source_analysis(data)

        print("\n7. Gender-based Financial Behavior Analysis")
        gender_financial_analysis(data)

        print("\n8. Investment Behavior Analysis")
        investment_behavior_analysis(data)

        print("\n9. Financial Challenges Analysis")
        financial_challenges_analysis(data)

        print("\n10. Financial Confidence Analysis")
        financial_confidence_analysis(data)

        print(f"\nAll plots have been saved in the '{plot_folder}' folder.")

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Convert output to image
    with open(output_file, 'r') as f:
        output_text = f.read()

    output_image = f'{plot_folder}/analysis_output.png'
    text_to_image(output_text, output_image)

    print(f"Analysis output has been saved as an image: {output_image}")