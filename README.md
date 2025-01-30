# Finance Tracker 💰

A comprehensive financial analysis and tracking system that helps analyze spending patterns, financial literacy, and provides insights through data visualization and machine learning models.

## 📁 Project Structure

```
├── financial_analysis.py           # Core analysis functions
├── financial_analysis_app.py       # Main application interface
├── model.py                        # Primary ML model implementation
├── modelst.py                      # Student-specific model
├── stmodel.py                      # Statistical modeling
├── student_financial_analysis.py   # Student finance analysis
├── Modified_Cleaned_Data.csv       # Primary dataset
├── Modified_Cleaned_Data_edited.csv# Enhanced dataset
├── expense_analysis.png           # Expense visualization
├── feature_importance.png         # ML feature importance
├── gender_literacy_analysis.png   # Gender-based analysis
├── Ques.docx                      # Documentation
└── Ques-1.docx                    # Additional documentation
```

## 🎯 Project Features

### Core Functionality
- Expense tracking and categorization
- Income and spending pattern analysis
- Financial literacy assessment
- Gender-based financial behavior analysis
- Machine learning predictions
- Interactive data visualizations

### Analysis Components
1. **Expense Analysis**
   - Category-wise breakdown
   - Trend analysis
   - Anomaly detection

2. **Financial Literacy**
   - Education impact assessment
   - Demographic analysis
   - Behavioral patterns

3. **Predictive Models**
   - Spending forecasts
   - Risk assessment
   - Budget recommendations

## 🛠️ Technical Setup

### Prerequisites
```python
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit (for app interface)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kanizmadix/finance_tracker.git
cd finance_tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Analysis

1. Core Analysis:
```bash
python financial_analysis.py
```

2. Interactive App:
```bash
streamlit run financial_analysis_app.py
```

3. Student-Specific Analysis:
```bash
python student_financial_analysis.py
```

### Data Input

The system accepts financial data in CSV format with the following structure:
- Transaction details
- Amount
- Category
- Date
- Demographics
- Financial literacy indicators

## 📊 Visualizations

1. **Expense Analysis** (`expense_analysis.png`)
   - Category-wise spending distribution
   - Time-series analysis
   - Spending patterns

2. **Feature Importance** (`feature_importance.png`)
   - Key factors affecting financial behavior
   - Impact weights of different variables

3. **Gender Literacy Analysis** (`gender_literacy_analysis.png`)
   - Gender-based financial patterns
   - Literacy correlation analysis

## 📈 Analysis Features

### Financial Metrics
- Monthly spending patterns
- Budget adherence
- Savings rate
- Risk assessment

### Machine Learning Models
- Spending prediction
- Category classification
- Anomaly detection
- Risk scoring

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/Enhancement`)
3. Commit your changes (`git commit -m 'Add Enhancement'`)
4. Push to the branch (`git push origin feature/Enhancement`)
5. Open a Pull Request

## 📝 Documentation

Detailed documentation is available in:
- `Ques.docx`: Primary documentation
- `Ques-1.docx`: Supplementary information

## 🔒 Security

- Sensitive financial data is handled securely
- Personal information is encrypted
- Compliance with financial data regulations

## 📊 Reports

The system generates various reports:
1. Monthly Financial Summary
2. Spending Pattern Analysis
3. Budget Performance
4. Risk Assessment Report

## 🎯 Future Enhancements

- Real-time transaction tracking
- Mobile app integration
- Advanced predictive analytics
- Investment portfolio analysis
- Multi-currency support

## 📫 Contact

Project Maintainer: KANISHK S
GitHub: [@kanizmadix](https://github.com/kanizmadix)

## 📜 License

This project is licensed under the MIT License - see the LICENSE.md file for details

---
⭐ If you find this tool helpful, please star the repository!

**Note**: This tool is designed for educational and personal use. For professional financial advice, please consult with qualified financial advisors.
