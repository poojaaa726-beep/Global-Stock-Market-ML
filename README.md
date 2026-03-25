# 📈 Global Stock Market — Machine Learning Project

An interactive web application built with **Streamlit** that applies Machine Learning techniques on a Global Stock Market dataset spanning **3 years (Jan 2023 – Mar 2026)**.

---

## 🌐 Live Demo
👉 [Click here to view the app](https://global-stock-market-ml.streamlit.app)

---

## 📊 Dataset Overview
| Detail | Info |
|---|---|
| Total Records | 90,040 |
| Companies | 200 |
| Countries | 10 |
| Date Range | Jan 2023 – Mar 2026 |
| Sectors | Multiple (Tech, Finance, Energy, etc.) |

---

## 🧠 5 Problem Statements

### PS1 — Stock Price Prediction
- **Goal:** Predict the next day's closing price
- **Models:** Linear Regression, Random Forest, Gradient Boosting
- **Metrics:** MAE, RMSE, R² Score

### PS2 — Buy / Sell Signal Classification
- **Goal:** Classify each trading day as BUY (1) or SELL (0)
- **Models:** Logistic Regression, Decision Tree, Random Forest
- **Metrics:** AUC Score, Accuracy, Confusion Matrix, ROC Curve

### PS3 — Cross-Market Performance Analysis
- **Goal:** Understand how sectors perform across countries
- **Methods:** K-Means Clustering, Heatmap, Box Plot, PCA

### PS4 — Investor Sentiment Analysis
- **Goal:** Check if BUY/SELL investor flow predicts future price
- **Methods:** Correlation Analysis, Rolling Window, RF Regression

### PS5 — Portfolio Optimization
- **Goal:** Build the best risk-adjusted investment portfolio
- **Methods:** Monte Carlo Simulation, Efficient Frontier, Sharpe Ratio

---

## 🛠️ Tech Stack
- **Python 3.10**
- **Streamlit** — Web app framework
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Machine learning models
- **Matplotlib & Seaborn** — Data visualization
- **SciPy** — Portfolio optimization

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/ankitlohan432-sketch/StockMarketML.git
cd StockMarketML
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
StockMarketML/
├── app.py                                  # Main Streamlit app
├── Global_Stock_Market_Master_Dataset.xlsx # Dataset
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 👤 Author
**Ankit Lohan**  
📧 Connect on [LinkedIn](https://linkedin.com)  
🌐 Portfolio: [your-portfolio-link.com](https://your-portfolio-link.com)

---

⭐ If you found this project useful, give it a star on GitHub!
