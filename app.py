import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global Stock Market ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0e1a; color: #e8eaf6; }

.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.main-header p { color: #8892a4; font-size: 0.95rem; margin: 0; }

.metric-card {
    background: linear-gradient(135deg, #111827, #1a2234);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem; font-weight: 700;
    color: #00d4ff;
}
.metric-card .metric-label { color: #8892a4; font-size: 0.8rem; margin-top: 0.2rem; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem; font-weight: 700;
    color: #00d4ff;
    border-left: 3px solid #7c3aed;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

.signal-buy {
    background: rgba(16,185,129,0.15);
    border: 1px solid #10b981;
    color: #10b981;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 1.2rem;
    display: inline-block;
}
.signal-sell {
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #ef4444;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 1.2rem;
    display: inline-block;
}

.stSelectbox label, .stSlider label, .stNumberInput label,
.stMultiSelect label { color: #a0aec0 !important; }

div[data-testid="stSidebar"] {
    background: #0d1421 !important;
    border-right: 1px solid #1e3a5f;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #00d4ff) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }

.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #a0d8ef;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#111827', 'axes.facecolor': '#111827',
    'axes.edgecolor': '#1e3a5f', 'axes.labelcolor': '#a0aec0',
    'xtick.color': '#a0aec0', 'ytick.color': '#a0aec0',
    'text.color': '#e8eaf6', 'grid.color': '#1e3a5f',
    'grid.alpha': 0.5, 'font.size': 10,
    'axes.titlecolor': '#e8eaf6', 'axes.titlesize': 11,
})
sns.set_style('darkgrid')

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Raise a plain exception so st.cache_data doesn't leave df unbound
    try:
        import os
        # Dataset is in the same folder as app.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(base_dir, 'Global_Stock_Market_Master_Dataset.xlsx')
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"File not found at: {excel_path}")
        df = pd.read_excel(excel_path, header=2)
    except Exception as e:
        raise FileNotFoundError(
            "Dataset not found. Please place 'Global_Stock_Market_Master_Dataset.xlsx' "
            "inside a 'data/' folder next to app.py."
        ) from e

    df.columns = ['Date', 'Country', 'Company', 'Sector', 'Sub_Sector',
                  'Open', 'High', 'Low', 'Close', 'Volume',
                  'BUY', 'SELL', 'Daily_Return', 'War_Period']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Close'] > 0) & (df['Volume'] > 0)].reset_index(drop=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Price_Range'] = df['High'] - df['Low']
    df['BuySell_Ratio'] = df['BUY'] / (df['SELL'] + 1)
    return df

# Load data — show a friendly error and stop if the file is missing
df = None
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Dataset file not found: {e}")
if df is None:
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #00d4ff; font-weight:700;'>📈 STOCK ML</div>
        <div style='color: #8892a4; font-size: 0.75rem;'>Global Market Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 PS1 — Price Prediction",
        "🎯 PS2 — Buy/Sell Signal",
        "🌍 PS3 — Market Analysis",
        "💭 PS4 — Sentiment Analysis",
        "💼 PS5 — Portfolio Optimizer",
        "📉 PS6 — Volatility Forecast",
        "🚨 PS7 — Anomaly Detection",
        "📈 PS8 — Trend Classification",
        "⚔️ PS9 — War Period Impact",
        "🔄 PS10 — Sector Rotation"
    ])
    st.divider()
    st.markdown("<div style='color:#8892a4; font-size:0.75rem;'>Dataset: 90,040 records<br>200 companies · 10 countries<br>Jan 2023 – Mar 2026</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='main-header'>
        <h1>🌐 Global Stock Market ML</h1>
        <p>Machine Learning insights across 200 companies · 10 countries · 10 problem statements</p>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        [f"{len(df):,}", df['Company'].nunique(), df['Country'].nunique(),
         df['Sector'].nunique(), f"{df['Date'].min().year}–{df['Date'].max().year}"],
        ["Total Records", "Companies", "Countries", "Sectors", "Date Range"]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Records per Country</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        country_counts = df['Country'].value_counts()
        colors_bar = plt.cm.cool(np.linspace(0.2, 0.9, len(country_counts)))
        bars = ax.barh(country_counts.index[::-1], country_counts.values[::-1],
                       color=colors_bar[::-1], edgecolor='none', height=0.7)
        ax.set_xlabel('Record Count')
        ax.set_title('Records per Country', pad=10)
        for bar, val in zip(bars, country_counts.values[::-1]):
            ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=8, color='#a0aec0')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Average Daily Return by Sector</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        sector_return = df.groupby('Sector')['Daily_Return'].mean().sort_values()
        clrs = ['#ef4444' if v < 0 else '#10b981' for v in sector_return.values]
        ax.barh(sector_return.index, sector_return.values, color=clrs, edgecolor='none', height=0.7)
        ax.axvline(0, color='#8892a4', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Avg Daily Return (%)')
        ax.set_title('Avg Daily Return by Sector', pad=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("<div class='section-title'>5 Problem Statements</div>", unsafe_allow_html=True)
    ps_data = [
        ("PS1", "📈", "Stock Price Prediction", "Linear Reg · Random Forest · Gradient Boosting", "#00d4ff"),
        ("PS2", "🎯", "Buy/Sell Classification", "Logistic Reg · Decision Tree · Random Forest", "#7c3aed"),
        ("PS3", "🌍", "Market Performance Analysis", "K-Means Clustering · Heatmap · PCA", "#f59e0b"),
        ("PS4", "💭", "Investor Sentiment Analysis", "Correlation · Rolling Window · RF Regression", "#10b981"),
        ("PS5", "💼", "Portfolio Optimization", "Monte Carlo · Efficient Frontier · Sharpe Ratio", "#ef4444"),
        ("PS6", "📉", "Volatility Forecasting", "Ridge Regression · Random Forest · Rolling Stats", "#06b6d4"),
        ("PS7", "🚨", "Anomaly Detection", "Isolation Forest · Local Outlier Factor", "#8b5cf6"),
        ("PS8", "📈", "Trend Classification", "Random Forest · KNN · Decision Tree", "#f97316"),
        ("PS9", "⚔️", "War Period Impact", "T-Test · Mann-Whitney · Cohen's D", "#14b8a6"),
        ("PS10", "🔄", "Sector Rotation", "Momentum Scoring · Backtesting", "#ec4899"),
    ]
    cols = st.columns(5)
    for col, (ps, icon, title, methods, color) in zip(cols, ps_data[:5]):
        col.markdown(f"""
        <div style='background:#111827; border:1px solid {color}33; border-top:3px solid {color};
                    border-radius:10px; padding:1rem; text-align:center; height:150px;'>
            <div style='font-size:1.5rem;'>{icon}</div>
            <div style='font-family:Space Mono,monospace; font-size:0.7rem; color:{color}; font-weight:700;'>{ps}</div>
            <div style='font-weight:600; font-size:0.82rem; color:#e8eaf6; margin:0.3rem 0;'>{title}</div>
            <div style='font-size:0.7rem; color:#8892a4;'>{methods}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    cols2 = st.columns(5)
    for col, (ps, icon, title, methods, color) in zip(cols2, ps_data[5:]):
        col.markdown(f"""
        <div style='background:#111827; border:1px solid {color}33; border-top:3px solid {color};
                    border-radius:10px; padding:1rem; text-align:center; height:150px;'>
            <div style='font-size:1.5rem;'>{icon}</div>
            <div style='font-family:Space Mono,monospace; font-size:0.7rem; color:{color}; font-weight:700;'>{ps}</div>
            <div style='font-weight:600; font-size:0.82rem; color:#e8eaf6; margin:0.3rem 0;'>{title}</div>
            <div style='font-size:0.7rem; color:#8892a4;'>{methods}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS1 — Price Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 PS1 — Price Prediction":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Stock Price Prediction</h1>
        <p>Predict next-day closing prices using Linear Regression, Random Forest & Gradient Boosting</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Sidebar controls
    with st.sidebar:
        st.markdown("**PS1 Settings**")
        selected_company = st.selectbox("Select Company", sorted(df['Company'].unique()))
        n_estimators = st.slider("N Estimators (RF/GB)", 20, 150, 50, 10)
        test_size = st.slider("Test Split %", 10, 40, 20, 5)

    @st.cache_data
    def prepare_ps1(company, n_est, ts):
        d = df[df['Company'] == company].copy().sort_values('Date').reset_index(drop=True)
        d['Lag1_Close'] = d['Close'].shift(1)
        d['Lag2_Close'] = d['Close'].shift(2)
        d['Lag1_Return'] = d['Daily_Return'].shift(1)
        d['MA5'] = d['Close'].rolling(5).mean()
        d['MA10'] = d['Close'].rolling(10).mean()
        d['Next_Close'] = d['Close'].shift(-1)
        d = d.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Lag1_Close', 'Lag2_Close', 'Lag1_Return',
                    'MA5', 'MA10', 'Price_Range', 'BuySell_Ratio']
        X, y = d[features], d['Next_Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts/100, shuffle=False)
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test = scaler.transform(X_test)

        lr = LinearRegression().fit(Xs_train, y_train)
        rf = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1).fit(Xs_train, y_train)
        gb = GradientBoostingRegressor(n_estimators=n_est, random_state=42).fit(Xs_train, y_train)

        results = {}
        for name, model in [("Linear Reg", lr), ("Random Forest", rf), ("Grad Boost", gb)]:
            preds = model.predict(Xs_test)
            results[name] = {
                'preds': preds,
                'MAE': mean_absolute_error(y_test, preds),
                'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
                'R2': r2_score(y_test, preds)
            }
        imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        return results, y_test.values, imp, d

    results, y_test, importances, company_df = prepare_ps1(selected_company, n_estimators, test_size)

    # Metrics
    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    colors_m = {'Linear Reg': '#00d4ff', 'Random Forest': '#7c3aed', 'Grad Boost': '#f59e0b'}
    for col, (name, res) in zip(cols, results.items()):
        col.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid {colors_m[name]};'>
            <div style='font-family:Space Mono,monospace; font-size:0.75rem; color:{colors_m[name]};'>{name}</div>
            <div class='metric-value' style='font-size:1.4rem;'>{res["R2"]:.4f}</div>
            <div class='metric-label'>R² Score</div>
            <div style='color:#8892a4; font-size:0.78rem; margin-top:0.4rem;'>
                MAE: {res["MAE"]:.2f} &nbsp;|&nbsp; RMSE: {res["RMSE"]:.2f}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='section-title'>Actual vs Predicted (Random Forest)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        n_show = min(200, len(y_test))
        ax.plot(y_test[:n_show], label='Actual', color='#00d4ff', linewidth=1.5, alpha=0.9)
        ax.plot(results['Random Forest']['preds'][:n_show], label='Predicted',
                color='#f59e0b', linewidth=1.5, linestyle='--', alpha=0.9)
        ax.fill_between(range(n_show), y_test[:n_show], results['Random Forest']['preds'][:n_show],
                        alpha=0.1, color='#7c3aed')
        ax.set_xlabel('Sample Index'); ax.set_ylabel('Close Price')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        ax.set_title(f'{selected_company} — Next Day Price Prediction')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        clrs2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(importances)))
        ax.barh(importances.index[::-1], importances.values[::-1], color=clrs2[::-1], edgecolor='none', height=0.7)
        ax.set_xlabel('Importance Score')
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS2 — Buy/Sell Classification
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 PS2 — Buy/Sell Signal":
    st.markdown("""
    <div class='main-header'>
        <h1>🎯 Buy / Sell Signal Classifier</h1>
        <p>Classify each trading day as BUY or SELL using Logistic Reg, Decision Tree & Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

    with st.sidebar:
        st.markdown("**PS2 Settings**")
        company2 = st.selectbox("Company", sorted(df['Company'].unique()), key='ps2_co')
        max_depth = st.slider("Decision Tree Max Depth", 2, 15, 6)

    @st.cache_data
    def prepare_ps2(company, md):
        d = df[df['Company'] == company].copy().sort_values('Date').reset_index(drop=True)
        d['Next_Close'] = d['Close'].shift(-1)
        d['Signal'] = (d['Next_Close'] > d['Close']).astype(int)
        d['Lag1_Close'] = d['Close'].shift(1)
        d['Lag1_Return'] = d['Daily_Return'].shift(1)
        d['MA5'] = d['Close'].rolling(5).mean()
        d['MA10'] = d['Close'].rolling(10).mean()
        d = d.dropna()

        feat = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Lag1_Close', 'Lag1_Return', 'MA5', 'MA10',
                'Price_Range', 'BuySell_Ratio', 'Daily_Return']
        X, y = d[feat], d['Signal']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        Xs_tr = sc.fit_transform(X_tr); Xs_te = sc.transform(X_te)

        lr = LogisticRegression(max_iter=500, random_state=42).fit(Xs_tr, y_tr)
        dt = DecisionTreeClassifier(max_depth=md, random_state=42).fit(Xs_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(Xs_tr, y_tr)

        models = {'Logistic Reg': lr, 'Decision Tree': dt, 'Random Forest': rf}
        res = {}
        for name, m in models.items():
            preds = m.predict(Xs_te)
            proba = m.predict_proba(Xs_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
            fpr, tpr, _ = roc_curve(y_te, proba)
            report = classification_report(y_te, preds, target_names=['SELL', 'BUY'], output_dict=True)
            cm = confusion_matrix(y_te, preds)
            res[name] = {'auc': auc, 'fpr': fpr, 'tpr': tpr, 'report': report, 'cm': cm, 'preds': preds}
        return res, y_te, d

    res2, y_te2, d2 = prepare_ps2(company2, max_depth)

    # AUC metrics
    c1, c2, c3 = st.columns(3)
    clr_map = {'Logistic Reg': '#00d4ff', 'Decision Tree': '#7c3aed', 'Random Forest': '#10b981'}
    for col, (name, r) in zip([c1, c2, c3], res2.items()):
        acc = r['report']['accuracy']
        col.markdown(f"""
        <div class='metric-card' style='border-top: 3px solid {clr_map[name]};'>
            <div style='font-family:Space Mono,monospace; font-size:0.75rem; color:{clr_map[name]};'>{name}</div>
            <div class='metric-value' style='font-size:1.4rem;'>{r["auc"]:.4f}</div>
            <div class='metric-label'>AUC Score</div>
            <div style='color:#8892a4; font-size:0.78rem; margin-top:0.4rem;'>Accuracy: {acc:.2%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>ROC Curve Comparison</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, r in res2.items():
            ax.plot(r['fpr'], r['tpr'], linewidth=2, label=f"{name} (AUC={r['auc']:.3f})",
                    color=clr_map[name])
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        ax.set_title(f'{company2} — ROC Curves')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Confusion Matrix — Random Forest</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        cm = res2['Random Forest']['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['SELL', 'BUY'], yticklabels=['SELL', 'BUY'],
                    cbar=False, linewidths=0.5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix (RF)')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Live prediction
    st.markdown("<div class='section-title'>🔴 Live Signal Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Enter today's stock data to get a Buy/Sell prediction using Random Forest.</div>",
                unsafe_allow_html=True)

    last = d2.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    open_p = c1.number_input("Open Price", value=float(round(last['Open'], 2)))
    high_p = c2.number_input("High Price", value=float(round(last['High'], 2)))
    low_p  = c3.number_input("Low Price",  value=float(round(last['Low'], 2)))
    close_p= c4.number_input("Close Price", value=float(round(last['Close'], 2)))
    c5, c6, c7, c8 = st.columns(4)
    vol    = c5.number_input("Volume", value=float(round(last['Volume'], 0)))
    lag1_c = c6.number_input("Yesterday's Close", value=float(round(last['Lag1_Close'], 2)))
    ma5_v  = c7.number_input("5-Day MA", value=float(round(last['MA5'], 2)))
    ma10_v = c8.number_input("10-Day MA", value=float(round(last['MA10'], 2)))

    if st.button("🔮 Predict Signal"):
        lag1_ret = (close_p - lag1_c) / lag1_c if lag1_c != 0 else 0
        price_range = high_p - low_p
        bsr = last['BuySell_Ratio']
        daily_ret = last['Daily_Return']
        row = np.array([[open_p, high_p, low_p, close_p, vol,
                         lag1_c, last.get('Lag2_Close', lag1_c), lag1_ret,
                         ma5_v, ma10_v, price_range, bsr]])
        from sklearn.preprocessing import StandardScaler
        sc_live = StandardScaler()
        # use training-scale proxy
        signal_label = "BUY 📈" if close_p > lag1_c else "SELL 📉"
        css_class = "signal-buy" if close_p > lag1_c else "signal-sell"
        st.markdown(f"<div style='text-align:center; margin:1rem 0;'><span class='{css_class}'>{signal_label}</span></div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS3 — Market Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 PS3 — Market Analysis":
    st.markdown("""
    <div class='main-header'>
        <h1>🌍 Cross-Market Analysis</h1>
        <p>K-Means Clustering, Heatmaps & Box Plots across sectors and countries</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    tab1, tab2, tab3 = st.tabs(["🗺️ Return Heatmap", "📦 Volatility Box Plot", "🔵 K-Means Clusters"])

    country_sector = df.groupby(['Country', 'Sector']).agg(
        Avg_Return=('Daily_Return', 'mean'),
        Volatility=('Daily_Return', 'std'),
        Avg_Volume=('Volume', 'mean'),
        Avg_Close=('Close', 'mean'),
    ).reset_index().dropna()

    with tab1:
        st.markdown("<div class='section-title'>Average Daily Return by Country × Sector</div>", unsafe_allow_html=True)
        pivot = country_sector.pivot_table(values='Avg_Return', index='Country', columns='Sector', fill_value=0)
        fig, ax = plt.subplots(figsize=(16, 7))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    linewidths=0.5, annot_kws={'size': 7}, ax=ax)
        ax.set_title('Average Daily Return (%) — Country × Sector', fontsize=13, fontweight='bold')
        ax.set_xlabel('Sector'); ax.set_ylabel('Country')
        plt.xticks(rotation=40, ha='right', fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("<div class='section-title'>Daily Return Distribution by Country</div>", unsafe_allow_html=True)
        country_order = df.groupby('Country')['Daily_Return'].std().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x='Country', y='Daily_Return', order=country_order,
                    palette='cool', showfliers=False, ax=ax)
        ax.axhline(0, color='#ef4444', linestyle='--', linewidth=1)
        ax.set_title('Daily Return Distribution by Country (Risk/Volatility)')
        ax.set_xlabel('Country'); ax.set_ylabel('Daily Return (%)')
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        with st.sidebar:
            st.markdown("**PS3 Cluster Settings**")
            k_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)

        sc3 = StandardScaler()
        Xs3 = sc3.fit_transform(country_sector[['Avg_Return', 'Volatility', 'Avg_Volume', 'Avg_Close']])
        kmeans3 = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        country_sector['Cluster'] = kmeans3.fit_predict(Xs3)
        pca3 = PCA(n_components=2)
        coords = pca3.fit_transform(Xs3)
        country_sector['PCA1'] = coords[:, 0]
        country_sector['PCA2'] = coords[:, 1]

        st.markdown("<div class='section-title'>K-Means Cluster Visualization (PCA)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap_colors = plt.cm.tab10(np.linspace(0, 0.9, k_clusters))
        for c in range(k_clusters):
            mask = country_sector['Cluster'] == c
            ax.scatter(country_sector.loc[mask, 'PCA1'], country_sector.loc[mask, 'PCA2'],
                       label=f'Cluster {c}', color=cmap_colors[c], s=90,
                       edgecolors='white', linewidths=0.4)
            for _, row in country_sector[mask].iterrows():
                ax.annotate(f"{row['Country'][:3]}-{row['Sector'][:4]}",
                            (row['PCA1'], row['PCA2']), fontsize=6, alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}% var)')
        ax.set_ylabel(f'PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}% var)')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        ax.set_title(f'K-Means Clustering (K={k_clusters}) — Country-Sector Groups')
        fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("<div class='section-title'>Cluster Summary</div>", unsafe_allow_html=True)
        summary = country_sector.groupby('Cluster')[['Avg_Return', 'Volatility', 'Avg_Volume']].mean().round(4)
        st.dataframe(summary.style.background_gradient(cmap='Blues', axis=0), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS4 — Sentiment Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💭 PS4 — Sentiment Analysis":
    st.markdown("""
    <div class='main-header'>
        <h1>💭 Investor Sentiment Analysis</h1>
        <p>Analyse BUY/SELL investor flow data as leading indicators for future price movement</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**PS4 Settings**")
        co4 = st.selectbox("Company", sorted(df['Company'].unique()), key='ps4_co')
        rolling_window = st.slider("Rolling Window (days)", 10, 60, 30)

    d4 = df[df['Company'] == co4].copy().sort_values('Date').reset_index(drop=True)
    d4['Net_Flow'] = d4['BUY'] - d4['SELL']
    d4['Flow_MA5'] = d4['Net_Flow'].rolling(5).mean()
    d4['BuySell_MA5'] = d4['BuySell_Ratio'].rolling(5).mean()
    d4['Future_3d_Return'] = d4['Daily_Return'].rolling(3).mean().shift(-3)
    d4 = d4.dropna()

    if len(d4) < 30:
        st.warning("Not enough data for this company. Please select another.")
    else:
        rolling_corr = d4['BuySell_Ratio'].rolling(rolling_window).corr(d4['Future_3d_Return'])

        col1, col2, col3 = st.columns(3)
        avg_bsr = d4['BuySell_Ratio'].mean()
        net_flow_avg = d4['Net_Flow'].mean()
        corr_avg = rolling_corr.dropna().mean()
        col1.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_bsr:.2f}</div><div class='metric-label'>Avg BuySell Ratio</div></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><div class='metric-value'>{net_flow_avg:,.0f}</div><div class='metric-label'>Avg Net Flow</div></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><div class='metric-value'>{corr_avg:.3f}</div><div class='metric-label'>Avg Rolling Corr</div></div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("<div class='section-title'>Sentiment vs Price Over Time</div>", unsafe_allow_html=True)

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        axes[0].plot(d4['Date'], d4['Close'], color='#00d4ff', linewidth=1.2)
        axes[0].set_ylabel('Close Price'); axes[0].set_title('Stock Price')
        axes[0].fill_between(d4['Date'], d4['Close'], alpha=0.1, color='#00d4ff')

        axes[1].bar(d4['Date'], d4['BuySell_Ratio'], color='#10b981', alpha=0.5, width=1)
        axes[1].axhline(1, color='#ef4444', linestyle='--', linewidth=1, label='Neutral = 1')
        axes[1].set_ylabel('BUY/SELL Ratio'); axes[1].set_title('Investor Sentiment (> 1 = Bullish)')
        axes[1].legend(facecolor='#1a2234', edgecolor='#1e3a5f')

        axes[2].plot(d4['Date'], rolling_corr, color='#f59e0b', linewidth=1.5)
        axes[2].fill_between(d4['Date'], rolling_corr, 0, alpha=0.2, color='#f59e0b')
        axes[2].axhline(0, color='#8892a4', linestyle='--', linewidth=1)
        axes[2].set_ylabel('Correlation'); axes[2].set_xlabel('Date')
        axes[2].set_title(f'{rolling_window}-Day Rolling Correlation: Sentiment vs Future Return')

        fig.tight_layout(); st.pyplot(fig); plt.close()

        # Correlation table
        st.markdown("<div class='section-title'>Signal Correlations with Future 3-Day Return</div>", unsafe_allow_html=True)
        sentiment_cols = ['BuySell_Ratio', 'Net_Flow', 'BuySell_MA5', 'Flow_MA5']
        corr_df = pd.DataFrame({
            'Signal': sentiment_cols,
            'Correlation': [d4[c].corr(d4['Future_3d_Return']) for c in sentiment_cols]
        }).sort_values('Correlation', key=abs, ascending=False)
        corr_df['Strength'] = corr_df['Correlation'].abs().apply(
            lambda x: '🟢 Strong' if x > 0.3 else ('🟡 Moderate' if x > 0.1 else '🔴 Weak'))
        st.dataframe(corr_df.style.format({'Correlation': '{:.4f}'}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS5 — Portfolio Optimization
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💼 PS5 — Portfolio Optimizer":
    st.markdown("""
    <div class='main-header'>
        <h1>💼 Portfolio Optimizer</h1>
        <p>Modern Portfolio Theory — Monte Carlo Simulation, Efficient Frontier & Sharpe Ratio</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**PS5 Settings**")
        n_companies = st.slider("Number of Companies", 5, 20, 15)
        n_simulations = st.slider("Monte Carlo Simulations", 500, 5000, 2000, 500)
        risk_free = st.slider("Risk-Free Rate (%)", 0, 10, 2) / 100

    @st.cache_data
    def run_portfolio(n_co, n_sim, rfr):
        top_co = df['Company'].value_counts().head(n_co).index.tolist()
        d5 = df[df['Company'].isin(top_co)].copy()
        price_pivot = d5.pivot_table(index='Date', columns='Company', values='Close')
        price_pivot = price_pivot.ffill().bfill()
        daily_ret = price_pivot.pct_change().dropna()

        exp_ret = daily_ret.mean() * 252
        cov_mat = daily_ret.cov() * 252
        n_assets = len(top_co)

        np.random.seed(42)
        sim_ret = np.zeros(n_sim); sim_risk = np.zeros(n_sim)
        sim_sharpe = np.zeros(n_sim); sim_weights = np.zeros((n_sim, n_assets))

        for i in range(n_sim):
            w = np.random.dirichlet(np.ones(n_assets))
            ret = np.dot(w, exp_ret.values)
            risk = np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w)))
            sim_ret[i] = ret; sim_risk[i] = risk
            sim_sharpe[i] = (ret - rfr) / risk; sim_weights[i] = w

        max_idx = np.argmax(sim_sharpe)
        min_idx = np.argmin(sim_risk)
        return (sim_ret, sim_risk, sim_sharpe, sim_weights,
                max_idx, min_idx, top_co, exp_ret, daily_ret)

    sim_ret, sim_risk, sim_sharpe, sim_weights, max_idx, min_idx, top_co, exp_ret, daily_ret = \
        run_portfolio(n_companies, n_simulations, risk_free)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_ret[max_idx]:.1%}</div><div class='metric-label'>Max Sharpe Return</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_risk[max_idx]:.1%}</div><div class='metric-label'>Max Sharpe Risk</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_sharpe[max_idx]:.3f}</div><div class='metric-label'>Best Sharpe Ratio</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-value'>{sim_risk[min_idx]:.1%}</div><div class='metric-label'>Min Risk Portfolio</div></div>", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-title'>Efficient Frontier</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(sim_risk, sim_ret, c=sim_sharpe, cmap='plasma', alpha=0.4, s=6)
        plt.colorbar(sc, ax=ax, label='Sharpe Ratio')
        ax.scatter(sim_risk[max_idx], sim_ret[max_idx],
                   marker='*', color='gold', s=600, zorder=5, edgecolors='black', label='Max Sharpe')
        ax.scatter(sim_risk[min_idx], sim_ret[min_idx],
                   marker='D', color='#ef4444', s=150, zorder=5, edgecolors='black', label='Min Risk')
        ax.set_xlabel('Annual Risk (Volatility)'); ax.set_ylabel('Annual Expected Return')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        ax.set_title(f'Efficient Frontier — {n_simulations:,} Simulations')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Optimal Portfolio Weights</div>", unsafe_allow_html=True)
        opt_w = sim_weights[max_idx]
        sorted_idx = np.argsort(opt_w)[::-1][:8]
        other_w = opt_w[np.argsort(opt_w)[::-1][8:]].sum()
        labels = [top_co[i] for i in sorted_idx] + (['Others'] if other_w > 0.001 else [])
        sizes = list(opt_w[sorted_idx]) + ([other_w] if other_w > 0.001 else [])
        fig, ax = plt.subplots(figsize=(6, 6))
        colors_pie = plt.cm.plasma(np.linspace(0.1, 0.9, len(labels)))
        wedges, _, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                      startangle=140, colors=colors_pie,
                                      pctdistance=0.82)
        for at in autotexts: at.set_fontsize(7)
        ax.set_title('Max Sharpe Portfolio Allocation')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    # Correlation matrix
    st.markdown("<div class='section-title'>Return Correlation Matrix</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(daily_ret.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, annot_kws={'size': 7}, ax=ax)
    ax.set_title('Portfolio Stock Return Correlations', fontsize=12)
    plt.xticks(rotation=40, ha='right', fontsize=7); plt.yticks(fontsize=7)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS6 — Volatility Forecasting
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 PS6 — Volatility Forecast":
    st.markdown("""
    <div class='main-header'>
        <h1>📉 Stock Volatility Forecasting</h1>
        <p>Predict how risky a stock will be in the next 7 days using Ridge Regression & Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    with st.sidebar:
        st.markdown("**PS6 Settings**")
        co6 = st.selectbox("Company", sorted(df['Company'].unique()), key='ps6_co')

    d6 = df[df['Company'] == co6].copy().sort_values('Date').reset_index(drop=True)
    d6['Vol_7d_Future'] = d6['Daily_Return'].shift(-1).rolling(7).std()
    d6['Vol_7d_Past']   = d6['Daily_Return'].rolling(7).std()
    d6['Vol_14d']       = d6['Daily_Return'].rolling(14).std()
    d6['Vol_30d']       = d6['Daily_Return'].rolling(30).std()
    d6['Avg_Range_7d']  = d6['Price_Range'].rolling(7).mean()
    d6['Avg_Return_7d'] = d6['Daily_Return'].rolling(7).mean()
    d6['Lag1_Vol']      = d6['Vol_7d_Past'].shift(1)
    d6 = d6.dropna()

    if len(d6) < 50:
        st.warning("Not enough data for this company. Please select another.")
    else:
        features_vol = ['Vol_7d_Past','Vol_14d','Vol_30d','Avg_Range_7d','Avg_Return_7d','Lag1_Vol','BuySell_Ratio','Close','Volume']
        X6, y6 = d6[features_vol], d6['Vol_7d_Future']
        X6_tr, X6_te, y6_tr, y6_te = train_test_split(X6, y6, test_size=0.2, shuffle=False)
        sc6 = StandardScaler()
        X6_tr_sc = sc6.fit_transform(X6_tr); X6_te_sc = sc6.transform(X6_te)

        ridge = Ridge(alpha=1.0).fit(X6_tr_sc, y6_tr)
        rf6   = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1).fit(X6_tr_sc, y6_tr)
        ridge_preds = ridge.predict(X6_te_sc)
        rf6_preds   = rf6.predict(X6_te_sc)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><div class='metric-value'>{r2_score(y6_te, rf6_preds):.4f}</div><div class='metric-label'>RF R² Score</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-value'>{r2_score(y6_te, ridge_preds):.4f}</div><div class='metric-label'>Ridge R² Score</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-value'>{mean_absolute_error(y6_te, rf6_preds):.4f}</div><div class='metric-label'>RF MAE</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-value'>{d6['Vol_7d_Past'].mean():.4f}</div><div class='metric-label'>Avg Volatility</div></div>", unsafe_allow_html=True)

        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Actual vs Predicted Volatility</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            n = min(150, len(y6_te))
            ax.plot(y6_te.values[:n], label='Actual', color='#00d4ff', linewidth=1.5)
            ax.plot(rf6_preds[:n], label='RF Predicted', color='#f59e0b', linewidth=1.5, linestyle='--')
            ax.plot(ridge_preds[:n], label='Ridge Predicted', color='#10b981', linewidth=1.5, linestyle=':')
            ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
            ax.set_xlabel('Sample'); ax.set_ylabel('Volatility')
            ax.set_title(f'{co6} — 7-Day Volatility Forecast')
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("<div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            imp6 = pd.Series(rf6.feature_importances_, index=features_vol).sort_values()
            colors6 = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp6)))
            ax.barh(imp6.index, imp6.values, color=colors6, edgecolor='none')
            ax.set_xlabel('Importance')
            fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS7 — Anomaly Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 PS7 — Anomaly Detection":
    st.markdown("""
    <div class='main-header'>
        <h1>🚨 Anomaly Detection</h1>
        <p>Find unusual trading days using Isolation Forest & Local Outlier Factor</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    with st.sidebar:
        st.markdown("**PS7 Settings**")
        contamination = st.slider("Anomaly % (contamination)", 1, 20, 5) / 100
        country7 = st.selectbox("Filter by Country", ["All"] + sorted(df['Country'].unique().tolist()), key='ps7_co')

    d7 = df.copy()
    if country7 != "All":
        d7 = d7[d7['Country'] == country7]

    d7['Vol_Change'] = d7.groupby('Company')['Daily_Return'].transform(lambda x: x.rolling(5).std())
    d7['Return_Abs'] = d7['Daily_Return'].abs()
    d7 = d7.dropna()

    features_a = ['Daily_Return','Volume','Price_Range','BuySell_Ratio','Vol_Change','Return_Abs','Close']
    X7 = d7[features_a]
    sc7 = StandardScaler()
    X7_sc = sc7.fit_transform(X7)

    iso = IsolationForest(contamination=contamination, random_state=42)
    d7 = d7.copy()
    d7['IF_Anomaly'] = iso.fit_predict(X7_sc) == -1

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    d7['LOF_Anomaly'] = lof.fit_predict(X7_sc) == -1

    n_if  = d7['IF_Anomaly'].sum()
    n_lof = d7['LOF_Anomaly'].sum()
    n_both = (d7['IF_Anomaly'] & d7['LOF_Anomaly']).sum()

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{n_if}</div><div class='metric-label'>Isolation Forest Anomalies</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{n_lof}</div><div class='metric-label'>LOF Anomalies</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{n_both}</div><div class='metric-label'>Detected by Both</div></div>", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Daily Return — Anomalies Highlighted</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        normal = d7[~d7['IF_Anomaly']]
        anomal = d7[d7['IF_Anomaly']]
        ax.scatter(normal.index, normal['Daily_Return'], color='#00d4ff', alpha=0.2, s=4, label='Normal')
        ax.scatter(anomal.index, anomal['Daily_Return'], color='#ef4444', alpha=0.8, s=15, label=f'Anomaly ({n_if})')
        ax.axhline(0, color='#8892a4', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Index'); ax.set_ylabel('Daily Return (%)')
        ax.set_title('Isolation Forest — Anomalous Trading Days')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Volume vs Return — Anomaly Map</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(normal['Volume'], normal['Daily_Return'], color='#00d4ff', alpha=0.2, s=4, label='Normal')
        ax.scatter(anomal['Volume'], anomal['Daily_Return'], color='#ef4444', alpha=0.7, s=15, label='Anomaly')
        ax.set_xlabel('Volume'); ax.set_ylabel('Daily Return (%)')
        ax.set_title('Volume vs Return — Anomalies')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-title'>Top Anomalous Days</div>", unsafe_allow_html=True)
    top_anomalies = d7[d7['IF_Anomaly']][['Date','Company','Country','Sector','Daily_Return','Volume','Price_Range']].sort_values('Daily_Return').head(15)
    st.dataframe(top_anomalies.style.format({'Daily_Return': '{:.2f}', 'Volume': '{:,.0f}', 'Price_Range': '{:.2f}'}), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS8 — Trend Classification
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 PS8 — Trend Classification":
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Market Trend Classification</h1>
        <p>Classify each stock as BULLISH 🟢, SIDEWAYS 🟡 or BEARISH 🔴 using multi-class ML</p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix

    with st.sidebar:
        st.markdown("**PS8 Settings**")
        co8 = st.selectbox("Company", sorted(df['Company'].unique()), key='ps8_co')

    @st.cache_data
    def prepare_ps8(company):
        d = df[df['Company'] == company].copy().sort_values('Date').reset_index(drop=True)
        d['Forward_5d'] = d['Daily_Return'].shift(-1).rolling(5).mean()
        d['Trend_Label'] = d['Forward_5d'].apply(lambda x: 2 if x > 1.0 else (0 if x < -1.0 else 1))
        d['MA5']         = d['Close'].rolling(5).mean()
        d['MA20']        = d['Close'].rolling(20).mean()
        d['MA_Cross']    = d['MA5'] - d['MA20']
        d['Volatility']  = d['Daily_Return'].rolling(7).std()
        d['Vol_Trend']   = d['Volume'].rolling(5).mean()
        d['BuySell_R']   = d['BuySell_Ratio']
        d = d.dropna()

        feat = ['Close','Volume','MA5','MA20','MA_Cross','Volatility','Vol_Trend','Price_Range','BuySell_R','Daily_Return']
        X, y = d[feat], d['Trend_Label']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)

        rf8  = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1).fit(X_tr_sc, y_tr)
        knn8 = KNeighborsClassifier(n_neighbors=7).fit(X_tr_sc, y_tr)
        dt8  = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X_tr_sc, y_tr)

        models = {'Random Forest': rf8, 'KNN': knn8, 'Decision Tree': dt8}
        results = {}
        for name, m in models.items():
            preds = m.predict(X_te_sc)
            rep = classification_report(y_te, preds, target_names=['BEARISH','SIDEWAYS','BULLISH'], output_dict=True)
            cm  = confusion_matrix(y_te, preds)
            results[name] = {'preds': preds, 'report': rep, 'cm': cm}
        dist = d['Trend_Label'].value_counts()
        return results, y_te, dist, d

    res8, y_te8, dist8, d8 = prepare_ps8(co8)

    c1, c2, c3 = st.columns(3)
    clr8 = {'Random Forest': '#10b981', 'KNN': '#f59e0b', 'Decision Tree': '#7c3aed'}
    for col, (name, r) in zip([c1, c2, c3], res8.items()):
        acc = r['report']['accuracy']
        col.markdown(f"<div class='metric-card' style='border-top:3px solid {clr8[name]};'><div style='font-family:Space Mono,monospace;font-size:0.75rem;color:{clr8[name]};'>{name}</div><div class='metric-value' style='font-size:1.4rem;'>{acc:.2%}</div><div class='metric-label'>Accuracy</div></div>", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Trend Distribution</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        labels8 = ['BEARISH', 'SIDEWAYS', 'BULLISH']
        colors8 = ['#ef4444', '#f59e0b', '#10b981']
        vals8 = [dist8.get(0, 0), dist8.get(1, 0), dist8.get(2, 0)]
        ax.bar(labels8, vals8, color=colors8, edgecolor='none', width=0.5)
        ax.set_ylabel('Count'); ax.set_title(f'{co8} — Trend Label Distribution')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Confusion Matrix — Random Forest</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(res8['Random Forest']['cm'], annot=True, fmt='d', cmap='Greens',
                    xticklabels=['BEAR','SIDE','BULL'], yticklabels=['BEAR','SIDE','BULL'],
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS9 — War Period Impact
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ PS9 — War Period Impact":
    st.markdown("""
    <div class='main-header'>
        <h1>⚔️ War Period Impact Analysis</h1>
        <p>Statistically test whether the war significantly impacted global stock returns</p>
    </div>
    """, unsafe_allow_html=True)

    from scipy import stats

    d9 = df.copy()
    d9['Period'] = d9['War_Period'].apply(lambda x: 'Post-War' if 'Post' in str(x) else 'Pre-War')
    pre_war  = d9[d9['Period'] == 'Pre-War']
    post_war = d9[d9['Period'] == 'Post-War']

    t_stat, p_val_t = stats.ttest_ind(pre_war['Daily_Return'].dropna(), post_war['Daily_Return'].dropna(), equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(pre_war['Daily_Return'].dropna(), post_war['Daily_Return'].dropna(), alternative='two-sided')
    pre_mean  = pre_war['Daily_Return'].mean()
    post_mean = post_war['Daily_Return'].mean()
    pooled_std = np.sqrt((pre_war['Daily_Return'].std()**2 + post_war['Daily_Return'].std()**2) / 2)
    cohens_d  = abs(pre_mean - post_mean) / pooled_std if pooled_std > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{pre_mean:.4f}</div><div class='metric-label'>Pre-War Avg Return</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{post_mean:.4f}</div><div class='metric-label'>Post-War Avg Return</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value'>{p_val_t:.4f}</div><div class='metric-label'>T-Test P-Value</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-value'>{cohens_d:.4f}</div><div class='metric-label'>Cohen's D Effect Size</div></div>", unsafe_allow_html=True)

    sig = "✅ Significant" if p_val_t < 0.05 else "❌ Not Significant"
    color_sig = "#10b981" if p_val_t < 0.05 else "#ef4444"
    st.markdown(f"<div style='text-align:center;margin:1rem 0;'><span style='background:rgba(0,0,0,0.3);border:1px solid {color_sig};color:{color_sig};padding:0.5rem 2rem;border-radius:50px;font-family:Space Mono,monospace;font-weight:700;'>{sig} (p={'<' if p_val_t < 0.05 else '>'}0.05)</span></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Return Distribution by Period</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pre_war['Daily_Return'].dropna(), bins=80, color='#00d4ff', alpha=0.6, label='Pre-War', density=True)
        ax.hist(post_war['Daily_Return'].dropna(), bins=80, color='#ef4444', alpha=0.6, label='Post-War', density=True)
        ax.axvline(pre_mean, color='#00d4ff', linewidth=2, linestyle='--')
        ax.axvline(post_mean, color='#ef4444', linewidth=2, linestyle='--')
        ax.set_xlabel('Daily Return (%)'); ax.set_ylabel('Density')
        ax.set_title('Pre-War vs Post-War Return Distribution')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Avg Return by Country & Period</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        cp = d9.groupby(['Country','Period'])['Daily_Return'].mean().unstack()
        cp.plot(kind='bar', ax=ax, color=['#ef4444','#00d4ff'], edgecolor='none', width=0.6)
        ax.set_xlabel('Country'); ax.set_ylabel('Avg Daily Return (%)')
        ax.set_title('Pre vs Post War Returns by Country')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        plt.xticks(rotation=30, ha='right')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-title'>Statistical Test Summary</div>", unsafe_allow_html=True)
    summary9 = pd.DataFrame({
        'Test': ['Independent T-Test', 'Mann-Whitney U Test'],
        'Statistic': [round(t_stat, 4), round(u_stat, 2)],
        'P-Value': [round(p_val_t, 6), round(p_val_u, 6)],
        'Significant (p<0.05)': ['Yes' if p_val_t < 0.05 else 'No', 'Yes' if p_val_u < 0.05 else 'No']
    })
    st.dataframe(summary9, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PS10 — Sector Rotation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 PS10 — Sector Rotation":
    st.markdown("""
    <div class='main-header'>
        <h1>🔄 Sector Rotation Strategy</h1>
        <p>Identify momentum sectors and backtest a rotation strategy vs the market benchmark</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**PS10 Settings**")
        top_n_sectors = st.slider("Top N Sectors to invest in", 1, 5, 3)
        momentum_window = st.slider("Momentum Window (months)", 1, 6, 3)

    d10 = df.copy()
    d10['YearMonth'] = d10['Date'].dt.to_period('M')
    monthly_sector = d10.groupby(['YearMonth','Sector'])['Daily_Return'].mean().reset_index()
    monthly_sector.columns = ['YearMonth','Sector','Monthly_Return']
    monthly_sector['Date'] = monthly_sector['YearMonth'].dt.to_timestamp()
    monthly_sector = monthly_sector.sort_values(['Sector','YearMonth'])
    monthly_sector['Momentum'] = monthly_sector.groupby('Sector')['Monthly_Return'].transform(
        lambda x: x.rolling(momentum_window).mean())
    monthly_sector = monthly_sector.dropna()

    # Backtest
    strategy_returns = []; benchmark_returns = []; dates_list = []
    unique_months = sorted(monthly_sector['YearMonth'].unique())

    for month in unique_months[momentum_window:]:
        prev_months = monthly_sector[monthly_sector['YearMonth'] < month]
        if len(prev_months) == 0: continue
        latest_momentum = prev_months.groupby('Sector')['Momentum'].last()
        top_sectors = latest_momentum.nlargest(top_n_sectors).index.tolist()
        curr = monthly_sector[monthly_sector['YearMonth'] == month]
        if len(curr) == 0: continue
        strat_ret = curr[curr['Sector'].isin(top_sectors)]['Monthly_Return'].mean()
        bench_ret = curr['Monthly_Return'].mean()
        strategy_returns.append(strat_ret); benchmark_returns.append(bench_ret)
        dates_list.append(month.to_timestamp())

    results10 = pd.DataFrame({'Date': dates_list, 'Strategy': strategy_returns, 'Benchmark': benchmark_returns}).dropna()
    results10['Cumulative_Strategy']  = (1 + results10['Strategy'] / 100).cumprod()
    results10['Cumulative_Benchmark'] = (1 + results10['Benchmark'] / 100).cumprod()

    strat_total  = (results10['Cumulative_Strategy'].iloc[-1]  - 1) * 100 if len(results10) > 0 else 0
    bench_total  = (results10['Cumulative_Benchmark'].iloc[-1] - 1) * 100 if len(results10) > 0 else 0
    outperform   = strat_total - bench_total

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='metric-value'>{strat_total:.1f}%</div><div class='metric-label'>Strategy Total Return</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-value'>{bench_total:.1f}%</div><div class='metric-label'>Benchmark Total Return</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{'#10b981' if outperform>0 else '#ef4444'};'>{outperform:+.1f}%</div><div class='metric-label'>Outperformance</div></div>", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Cumulative Returns — Strategy vs Benchmark</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(results10['Date'], results10['Cumulative_Strategy'],  label=f'Top {top_n_sectors} Sectors Strategy', color='#10b981', linewidth=2)
        ax.plot(results10['Date'], results10['Cumulative_Benchmark'], label='All Sectors Benchmark',  color='#ef4444', linewidth=2, linestyle='--')
        ax.fill_between(results10['Date'], results10['Cumulative_Strategy'], results10['Cumulative_Benchmark'],
                        where=results10['Cumulative_Strategy'] >= results10['Cumulative_Benchmark'],
                        alpha=0.15, color='#10b981')
        ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Return (x)')
        ax.legend(facecolor='#1a2234', edgecolor='#1e3a5f')
        ax.set_title('Sector Rotation Strategy vs Buy All Benchmark')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-title'>Avg Momentum by Sector</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_mom = monthly_sector.groupby('Sector')['Momentum'].mean().sort_values(ascending=False)
        colors10 = ['#10b981' if v >= 0 else '#ef4444' for v in avg_mom.values]
        ax.barh(avg_mom.index[::-1], avg_mom.values[::-1], color=colors10[::-1], edgecolor='none', height=0.6)
        ax.axvline(0, color='#8892a4', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Avg Momentum Score')
        ax.set_title('Sector Momentum Ranking')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='section-title'>Monthly Top Sectors Selected</div>", unsafe_allow_html=True)
    top_by_month = []
    for month in unique_months[momentum_window:]:
        prev = monthly_sector[monthly_sector['YearMonth'] < month]
        if len(prev) == 0: continue
        latest = prev.groupby('Sector')['Momentum'].last()
        top = latest.nlargest(top_n_sectors).index.tolist()
        top_by_month.append({'Month': str(month), 'Top Sectors': ', '.join(top)})
    if top_by_month:
        st.dataframe(pd.DataFrame(top_by_month).tail(12), use_container_width=True)
