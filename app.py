import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



# PAGE CONFIG

st.set_page_config(
    page_title="AI Based Mineral Trade Intelligence",
    page_icon="📊",
    layout="wide"
)


# CLEAN POLICY-STYLE UI

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
    color: #1f2937;
    font-family: 'Inter', sans-serif;
}
.section-header {
    font-size: 26px;
    font-weight: 700;
    color: #1e3a8a;
    border-left: 6px solid #2563eb;
    padding-left: 12px;
    margin-bottom: 18px;
}
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    margin-bottom: 20px;
}
.card h2 {
    color: #2563eb;
}
.insight {
    background: #eef2ff;
    border-left: 5px solid #4f46e5;
    padding: 16px;
    border-radius: 10px;
    font-size: 15px;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}
.stDownloadButton button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 8px;
}
.footer {
    text-align: center;
    margin-top: 40px;
    color: #6b7280;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

plt.style.use("seaborn-v0_8-darkgrid")


# LOAD DATA

@st.cache_data
def load_data():
    df_import = pd.read_csv("import_final.csv", encoding="latin1")
    df_export = pd.read_csv("export_final.csv", encoding="latin1")

    df_import["Year"] = pd.to_datetime(df_import["Year"], errors="coerce")
    df_export["Year"] = pd.to_datetime(df_export["Year"], errors="coerce")

    df_trade = pd.merge(df_import, df_export, on="Year", how="inner")
    df_trade["Trade_Balance"] = df_trade["Export"] - df_trade["Import"]

    return df_import, df_export, df_trade

df_import, df_export, df_trade = load_data()

@st.cache_data
def load_state_mapping():
    df = pd.read_csv("state_level_mineral_mapping.csv")
    return df

df_state = load_state_mapping()



# KPI VALUES

import_delta = df_import.iloc[-1]["Import"] - df_import.iloc[-2]["Import"]
export_delta = df_export.iloc[-1]["Export"] - df_export.iloc[-2]["Export"]
trade_balance = df_trade.iloc[-1]["Trade_Balance"]



# YEAR → QUARTER CONVERSION (SARIMA)

def yearly_to_quarterly(df, value_col):
    rows = []
    for _, row in df.iterrows():
        year = row["Year"].year
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            rows.append({
                "Date": f"{year}-{q}",
                value_col: row[value_col] / 4
            })
    qdf = pd.DataFrame(rows)
    qdf["Date"] = pd.PeriodIndex(qdf["Date"], freq="Q").to_timestamp()
    qdf.set_index("Date", inplace=True)
    return qdf

df_import_q = yearly_to_quarterly(df_import, "Import")


# TRADE PARTNER INTELLIGENCE (PART 3)
partner_intelligence = {
    "Copper": {
        "sources": ["Chile", "Peru"],
        "dependency": "High",
        "risk": "High"
    },
    "Graphite": {
        "sources": ["China", "Mozambique"],
        "dependency": "Medium",
        "risk": "Moderate"
    },
    "Lithium": {
        "sources": ["Australia", "Chile", "Argentina"],
        "dependency": "Very High",
        "risk": "Critical"
    }
}


# SIDEBAR

st.sidebar.title("⚙️ Controls")

model_type = st.sidebar.selectbox(
    "Forecast Model",
    ["ARIMA (Yearly)", "SARIMA (Quarterly)", "Hybrid (ARIMA + LSTM)"]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Years)",
    1, 5, 3
)

st.sidebar.markdown("---")
st.sidebar.caption("📊 Source: DGCI&S / IBM")
st.sidebar.caption("⚙️ Engine: statsmodels")


# HEADER

st.markdown("""
<div class="card">
<h1>🪨 AI Based Minerals Trade Intelligence Platform</h1>
<p>
Policy-grade analytics platform for forecasting India’s critical mineral
imports, exports, and trade balance.
</p>
</div>
""", unsafe_allow_html=True)


# ANOVA RESULTS (FROM NOTEBOOK – FIXED, VALIDATED)

# Import ANOVA (Copper vs Graphite)
anova_import = {
    "F": 23.81,
    "p": 0.00038
}

# Export ANOVA (Copper vs Graphite)
anova_export = {
    "F": 7.55,      
    "p": 0.01765     
}


# LSTM HELPER FUNCTIONS (HYBRID MODEL)


def create_lstm_data(series, lookback=2):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)


def train_lstm(series, forecast_steps, lookback=2):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_lstm_data(scaled, lookback)

    if len(X) < 5:
        return np.zeros(forecast_steps)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, activation="tanh", input_shape=(lookback, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=200, verbose=0)

    last_seq = scaled[-lookback:].reshape(1, lookback, 1)

    preds = []
    for _ in range(forecast_steps):
        p = model.predict(last_seq, verbose=0)[0][0]
        preds.append(p)
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1, 0] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds.flatten()



Feature_1, Feature_2, Feature_3, Feature_4, Feature_5, Feature_6, Feature_7, Feature_8, Feature_9 = st.tabs([
    " Overview",
    "Import Forecast",
    "Export Forecast",
    "Trade Balance",
    "Trade Partners",
    "ANOVA Validation",
    "Dependency Ratio",
    "Risk Index",
    "State_extraction"
    
])



# TAB 1 — OVERVIEW

with Feature_1:
    st.markdown('<div class="section-header">📌 Trade Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Imports", f"{df_import.iloc[-1]['Import']:.2f}", f"{import_delta:.2f}")
    c2.metric("Exports", f"{df_export.iloc[-1]['Export']:.2f}", f"{export_delta:.2f}")
    c3.metric("Trade Balance", f"{trade_balance:.2f}")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_import["Year"].dt.year, df_import["Import"], label="Import")
    ax.plot(df_export["Year"].dt.year, df_export["Export"], label="Export")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    <div class="insight">
    Rising imports and volatile exports indicate growing dependency
    on foreign mineral supply chains.
    </div>
    """, unsafe_allow_html=True)


# TAB 2 — IMPORT FORECAST

with Feature_2:
    st.markdown('<div class="section-header">📥 Import Forecast</div>', unsafe_allow_html=True)

    # ---------- ARIMA ----------
    if model_type == "ARIMA (Yearly)":
        model = ARIMA(df_import["Import"], order=(0,1,0))
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_horizon)

        years = range(
            df_import["Year"].dt.year.iloc[-1] + 1,
            df_import["Year"].dt.year.iloc[-1] + 1 + len(forecast)
        )

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_import["Year"].dt.year, df_import["Import"], label="Actual")
        ax.plot(years, forecast.values, label="ARIMA Forecast", linestyle="--", linewidth=2)
        ax.legend()
        st.pyplot(fig)

    #  SARIMA
    elif model_type == "SARIMA (Quarterly)":
        steps = forecast_horizon * 4
        model = SARIMAX(
            df_import_q["Import"],
            order=(1,1,1),
            seasonal_order=(1,1,1,4)
        )
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=steps)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_import_q.index, df_import_q["Import"], label="Actual")
        ax.plot(forecast.index, forecast.values, label="SARIMA Forecast", linestyle="--")
        ax.legend()
        st.pyplot(fig)

    #  HYBRID (ARIMA + LSTM) 
    else:
        #  ARIMA
        arima_model = ARIMA(df_import["Import"], order=(0,1,0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_horizon)

        #  Residuals
        residuals = arima_fit.resid

        #  LSTM on residuals
        lstm_forecast = train_lstm(residuals, forecast_horizon)

        #  Hybrid forecast
        hybrid_forecast = arima_forecast.values + lstm_forecast

        years = range(
            df_import["Year"].dt.year.iloc[-1] + 1,
            df_import["Year"].dt.year.iloc[-1] + 1 + forecast_horizon
        )

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_import["Year"].dt.year, df_import["Import"], label="Actual Import")
        ax.plot(years, hybrid_forecast, label="Hybrid Forecast (ARIMA + LSTM)", linestyle="--", linewidth=3)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        <div class="insight">
        Hybrid model combines ARIMA’s linear trend learning with LSTM’s ability
        to capture nonlinear trade shocks and policy disruptions.
        </div>
        """, unsafe_allow_html=True)



# EXPORT FORECAST

with Feature_3:
    st.markdown('<div class="section-header">📤 Export Forecast</div>', unsafe_allow_html=True)

    #  ARIMA 
    if model_type == "ARIMA (Yearly)":
        model = ARIMA(df_export["Export"], order=(1,0,1))
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_horizon)

        years = range(
            df_export["Year"].dt.year.iloc[-1] + 1,
            df_export["Year"].dt.year.iloc[-1] + 1 + len(forecast)
        )

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_export["Year"].dt.year, df_export["Export"], label="Actual Export")
        ax.plot(years, forecast.values, label="ARIMA Forecast", linestyle="--")
        ax.legend()
        st.pyplot(fig)

    #  SARIMA 
    elif model_type == "SARIMA (Quarterly)":
        steps = forecast_horizon * 4

        df_export_q = yearly_to_quarterly(df_export, "Export")

        model = SARIMAX(
            df_export_q["Export"],
            order=(1,1,1),
            seasonal_order=(1,1,1,4)
        )
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=steps)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_export_q.index, df_export_q["Export"], label="Actual Export")
        ax.plot(forecast.index, forecast.values, label="SARIMA Forecast", linestyle="--")
        ax.legend()
        st.pyplot(fig)

    #  HYBRID (ARIMA + LSTM) 
    else:
        arima_model = ARIMA(df_export["Export"], order=(1,0,1))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_horizon)

        residuals = arima_fit.resid
        lstm_forecast = train_lstm(residuals, forecast_horizon)

        hybrid_forecast = arima_forecast.values + lstm_forecast

        years = range(
            df_export["Year"].dt.year.iloc[-1] + 1,
            df_export["Year"].dt.year.iloc[-1] + 1 + forecast_horizon
        )

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_export["Year"].dt.year, df_export["Export"], label="Actual Export")
        ax.plot(years, hybrid_forecast, label="Hybrid Forecast (ARIMA + LSTM)", linestyle="--", linewidth=3)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        <div class="insight">
        Hybrid forecasting captures nonlinear export volatility and
        international demand shocks beyond classical ARIMA.
        </div>
        """, unsafe_allow_html=True)



# TRADE BALANCE

with Feature_4:
    st.markdown('<div class="section-header">⚖️ Trade Balance Forecast</div>', unsafe_allow_html=True)

    
    if model_type == "ARIMA (Yearly)":
        model = ARIMA(df_trade["Trade_Balance"], order=(1,0,1))
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_horizon)

        years = range(
            df_trade["Year"].dt.year.iloc[-1] + 1,
            df_trade["Year"].dt.year.iloc[-1] + 1 + len(forecast)
        )

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(
            df_trade["Year"].dt.year,
            df_trade["Trade_Balance"],
            label="Actual Trade Balance",
            linewidth=2
        )
        ax.plot(
            years,
            forecast.values,
            label="ARIMA Forecast",
            linestyle="--",
            linewidth=3
        )
        ax.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    #  SARIMA 
    elif model_type == "SARIMA (Quarterly)":
        steps = forecast_horizon * 4

        df_trade_q = yearly_to_quarterly(df_trade, "Trade_Balance")

        model = SARIMAX(
            df_trade_q["Trade_Balance"],
            order=(1,1,1),
            seasonal_order=(1,1,1,4)
        )
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=steps)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(
            df_trade_q.index,
            df_trade_q["Trade_Balance"],
            label="Actual Trade Balance",
            linewidth=2
        )
        ax.plot(
            forecast.index,
            forecast.values,
            label="SARIMA Forecast",
            linestyle="--",
            linewidth=3
        )
        ax.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    #  HYBRID (ARIMA + LSTM)
    else:
        arima_model = ARIMA(df_trade["Trade_Balance"], order=(1,0,1))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=forecast_horizon)

        residuals = arima_fit.resid
        lstm_forecast = train_lstm(residuals, forecast_horizon)

        hybrid_forecast = arima_forecast.values + lstm_forecast

        years = range(
            df_trade["Year"].dt.year.iloc[-1] + 1,
            df_trade["Year"].dt.year.iloc[-1] + 1 + forecast_horizon
        )

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_trade["Year"].dt.year, df_trade["Trade_Balance"], label="Actual Trade Balance")
        ax.plot(years, hybrid_forecast, label="Hybrid Forecast (ARIMA + LSTM)", linestyle="--", linewidth=3)
        ax.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

        st.markdown("""
        <div class="insight">
        Hybrid forecasting improves trade balance prediction by modeling
        nonlinear demand–supply imbalances and policy-driven shocks.
        </div>
        """, unsafe_allow_html=True)




# TRADE PARTNER INTELLIGENCE 

with Feature_5:
    st.markdown('<div class="section-header">🌍 Trade Partner Intelligence</div>', unsafe_allow_html=True)

    mineral = st.selectbox("Select Mineral", list(partner_intelligence.keys()))
    info = partner_intelligence[mineral]

    st.markdown(f"""
    <div class="card">
    <h3>{mineral}</h3>
    <p><b>Major Sources:</b> {", ".join(info["sources"])}</p>
    <p><b>Dependency Level:</b> {info["dependency"]}</p>
    <p><b>Supply Risk:</b> {info["risk"]}</p>
    </div>
    """, unsafe_allow_html=True)



# ANOVA VALIDATION (IMPORT + EXPORT)

with Feature_6:
    st.markdown('<div class="section-header">🧪 ANOVA Statistical Validation</div>', unsafe_allow_html=True)

    st.markdown("""
    **Objective:**  
    To statistically validate whether different critical minerals exhibit
    significantly different trade behavior, justifying mineral-specific
    forecasting and policy decisions.
    """)

    
    st.markdown("### 📥 Import ANOVA (Copper vs Graphite)")

    c1, c2 = st.columns(2)
    c1.metric("F-Statistic", f"{anova_import['F']}")
    c2.metric("P-Value", f"{anova_import['p']}")

    if anova_import["p"] < 0.05:
        st.success("✅ Statistically Significant Difference in Imports")
        st.markdown("""
        **Interpretation:**  
        Import patterns of Copper and Graphite differ significantly.  
        Mineral-specific import dependency and risk assessment is justified.
        """)
    else:
        st.warning("❌ No Significant Difference Detected in Imports")

    st.markdown("---")


    st.markdown("### 📤 Export ANOVA (Copper vs Graphite)")

    c3, c4 = st.columns(2)
    c3.metric("F-Statistic", f"{anova_export['F']}")
    c4.metric("P-Value", f"{anova_export['p']}")

    if anova_export["p"] < 0.05:
        st.success("✅ Statistically Significant Difference in Exports")
        st.markdown("""
        **Interpretation:**  
        Export performance varies significantly across minerals,  
        indicating differing global competitiveness and demand dynamics.
        """)
    else:
        st.warning("❌ No Significant Difference Detected in Exports")

    
    st.markdown("""
    <div class="insight">
    <b>Policy Insight:</b><br>
    ANOVA validation confirms that treating all minerals uniformly would
    mask critical differences. Mineral-wise forecasting and targeted policy
    intervention are statistically justified.
    </div>
    """, unsafe_allow_html=True)

# load the dependency_ratio_csv    

df_dependency = pd.read_csv("dependency_ratio_final.csv")



#  Supplier concentration proxy (clearly labelled)
# Assumption: Higher import dependency → higher supplier concentration
df_dependency["Supplier_Concentration"] = df_dependency["Dependency_Ratio"]


vol_df = (
    df_dependency
    .groupby("Mineral")["Dependency_Ratio"]
    .std()
    .reset_index()
    .rename(columns={"Dependency_Ratio": "Import_Volatility"})
)

df_dependency = df_dependency.merge(vol_df, on="Mineral", how="left")


df_dependency["Import_Volatility"] = df_dependency["Import_Volatility"].fillna(
    df_dependency["Import_Volatility"].mean()
)


df_dependency["Risk_Index"] = (
    0.5 * df_dependency["Dependency_Ratio"] +
    0.3 * df_dependency["Import_Volatility"] +
    0.2 * df_dependency["Supplier_Concentration"]
)


def classify_risk(x):
    if x >= 0.8:
        return "🚨 Critical"
    elif x >= 0.6:
        return "🔴 High"
    elif x >= 0.3:
        return "🟠 Moderate"
    else:
        return "🟢 Low"

df_dependency["Risk_Level"] = df_dependency["Risk_Index"].apply(classify_risk)




# DEPENDENCY RATIO DASHBOARD

with Feature_7:
    st.markdown('<div class="section-header">📉 Import Dependency Ratio</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Definition:</b><br>
    Dependency Ratio = Imports / (Imports + Domestic Production)
    <br>
    <i>Computed using IBM mineral production data.</i>
    </div>
    """, unsafe_allow_html=True)

    # Mineral selector
    mineral = st.selectbox(
        "Select Mineral",
        sorted(df_dependency["Mineral"].unique())
    )

    df_m = df_dependency[df_dependency["Mineral"] == mineral].sort_values("Year")

    
    latest_ratio = df_m.iloc[-1]["Dependency_Ratio"]

    col1, col2 = st.columns([1, 2])

    with col1:
        if latest_ratio >= 0.8:
            st.error(f"🔴 Critical Dependency: {latest_ratio:.2%}")
        elif latest_ratio >= 0.5:
            st.warning(f"🟠 High Dependency: {latest_ratio:.2%}")
        else:
            st.success(f"🟢 Moderate Dependency: {latest_ratio:.2%}")

    with col2:
        st.metric(
            "Latest Dependency Ratio",
            f"{latest_ratio:.2%}",
            delta=f"{(latest_ratio - df_m.iloc[-2]['Dependency_Ratio']):.2%}"
        )

    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df_m["Year"],
        df_m["Dependency_Ratio"] * 100,
        marker="o",
        linewidth=2
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Dependency Ratio (%)")
    ax.set_xlabel("Year")
    ax.set_title(f"{mineral} – Import Dependency Trend")

    ax.grid(True)
    st.pyplot(fig)

    #  POLICY INSIGHT 
    st.markdown(f"""
    <div class="insight">
    <b>Policy Insight:</b><br>
    India depends on imports for <b>{latest_ratio:.0%}</b> of its {mineral} requirement.
    Sustained high dependency indicates strategic vulnerability and supply-chain risk.
    </div>
    """, unsafe_allow_html=True)



#  CRITICAL MINERAL RISK INDEX

with Feature_8:
    st.markdown('<div class="section-header">⚠️ Critical Mineral Risk Index</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Risk Index Definition:</b><br>
    Composite score measuring India’s vulnerability to mineral supply shocks.
    <br><br>
    <b>Formula:</b><br>
    Risk = 0.5 × Dependency + 0.3 × Import Volatility + 0.2 × Supplier Concentration
    <br>
    <i>(Supplier concentration is a proxy due to data constraints)</i>
    </div>
    """, unsafe_allow_html=True)

    mineral = st.selectbox(
        "Select Mineral",
        sorted(df_dependency["Mineral"].unique()),
        key="risk_mineral"
    )

    df_r = df_dependency[df_dependency["Mineral"] == mineral].sort_values("Year")
    latest = df_r.iloc[-1]

    # ---------------- KPI CARDS ----------------
    c1, c2, c3 = st.columns(3)

    c1.metric("Dependency Ratio", f"{latest['Dependency_Ratio']:.2%}")
    c2.metric("Risk Index Score", f"{latest['Risk_Index']:.2f}")
    c3.metric("Risk Level", latest["Risk_Level"])

    # ---------------- TREND CHART ----------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df_r["Year"],
        df_r["Risk_Index"],
        marker="o",
        linewidth=2
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk Index")
    ax.set_xlabel("Year")
    ax.set_title(f"{mineral} – Critical Mineral Risk Trend")
    ax.grid(True)
    st.pyplot(fig)

    # ---------------- POLICY INSIGHT ----------------
    st.markdown(f"""
    <div class="insight">
    <b>Strategic Insight:</b><br>
    {mineral} currently falls under <b>{latest['Risk_Level']}</b> category.
    Sustained high risk indicates urgent need for domestic exploration,
    supplier diversification, and strategic stockpiling.
    </div>
    """, unsafe_allow_html=True)

FOCUS_MINERALS = [
    "Copper Ores & Conc",
    "Lithium",
    "Graphite(Natural)"
]


with Feature_9:
    st.markdown('<div class="section-header">🗺️ State-Level Mineral Extraction Mapping</div>', unsafe_allow_html=True)

    state = st.selectbox(
        "Select State",
        sorted(df_state["State"].unique())
    )

    df_s = df_state[df_state["State"] == state].sort_values("Year")

    if df_s.empty:
        st.warning("⚠️ No data available for selected state.")
        st.stop()

    latest = df_s.iloc[-1]

    #  KPI CARDS 
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Production Value (Cr ₹)", f"{df_s['Value_Cr'].sum():.0f}")
    c2.metric("Total GST Collection (Cr ₹)", f"{df_s['GST_Collection_Cr_INR'].sum():.0f}")
    c3.metric("State", state)

    #  OVERALL TREND 
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(df_s["Year"], df_s["Value_Cr"], label="Total Production", linewidth=2)
    ax.plot(df_s["Year"], df_s["GST_Collection_Cr_INR"], label="Total GST Activity", linewidth=2)

    ax.set_title(f"{state}: Production vs GST Mineral Activity")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    #  CRITICAL MINERALS 
    st.markdown("### 🔍 Critical Minerals Focus")

    df_focus = df_s[df_s["Mineral"].isin(FOCUS_MINERALS)]

    if df_focus.empty:
        st.info("No Copper / Lithium / Graphite activity recorded for this state.")
    else:
       
        summary = (
            df_focus
            .groupby("Mineral", as_index=False)
            .agg({
                "Value_Cr": "sum",
                "GST_Collection_Cr_INR": "sum"
            })
        )

        st.dataframe(
            summary.rename(columns={
                "Value_Cr": "Production Value (Cr ₹)",
                "GST_Collection_Cr_INR": "GST Collection (Cr ₹)"
            }),
            use_container_width=True
        )

        
        fig2, ax2 = plt.subplots(figsize=(10,4))

        for mineral in summary["Mineral"]:
            df_m = df_focus[df_focus["Mineral"] == mineral]
            ax2.plot(
                df_m["Year"],
                df_m["Value_Cr"],
                marker="o",
                label=mineral
            )

        ax2.set_title(f"{state}: Critical Mineral Production Trend")
        ax2.set_ylabel("Production Value (Cr ₹)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        st.pyplot(fig2)

    
    st.markdown(f"""
    <div class="insight">
    <b>Policy Insight:</b><br>
    {state} shows distinct extraction and GST behavior for
    <b>Copper, Lithium, and Graphite</b>.
    Differences between production value and GST activity
    indicate inter-state mineral movement, processing hubs,
    and value-chain fragmentation.
    </div>
    """, unsafe_allow_html=True)





# FOOTER

st.markdown("""
<div class="footer">
Built for IIT-ISM Dhanbad Hackathon • Team CRX • 2026
</div>
""", unsafe_allow_html=True)
