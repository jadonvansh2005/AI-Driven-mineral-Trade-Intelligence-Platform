🪨 AI-Based Mineral Trade Intelligence System

The AI-Based Mineral Trade Intelligence System is a policy-grade analytics platform designed to strengthen India’s critical mineral security.
It analyzes EXIM trade data, domestic mineral production, and state-level economic activity to deliver accurate forecasting, import dependency analysis, and risk assessment for key minerals such as Copper, Graphite, and Lithium.

Using statistical validation, time-series forecasting, and AI models, the system identifies vulnerable minerals, predicts future trade trends, and highlights state-level value-chain gaps.
The solution is delivered as an interactive Streamlit dashboard, making it accessible to policy makers, planners, and non-technical stakeholders.

🎯 Key Capabilities

Mineral-wise import/export trend analysis

Statistically validated modeling using ANOVA

AI-powered forecasting (ARIMA, SARIMA, Hybrid ARIMA + LSTM)

Import Dependency Ratio calculation

Critical Mineral Risk Index (Low → Critical)

State-level production vs value-chain mapping

Policy-ready interactive dashboard

🛠️ Technologies & Tools Used
Programming & Data Science

Python

NumPy

Pandas

Scikit-learn

Statsmodels

Machine Learning & Forecasting

ARIMA

SARIMA

LSTM (TensorFlow / Keras)

Hybrid ARIMA + LSTM

Visualization & Dashboard

Matplotlib

Streamlit

Data Sources

DGCI&S – Export–Import (EXIM) Trade Data

IBM – Mineral Production Data

GSI – Exploration & Reserve Information

GST Data – State-level aggregated proxy (policy-compliant)

🏁 Outcome

Identifies high-risk critical minerals

Improves forecast accuracy using hybrid AI models

Supports evidence-based mineral policy decisions

Highlights state-level infrastructure and logistics gaps

📊 Dataset Summary
Dataset	Source	Purpose
Import Data	DGCI&S	Trade inflow analysis
Export Data	DGCI&S	Trade outflow analysis
Production Data	IBM	Domestic supply
Exploration Data	GSI	Strategic context
GST Data	Synthetic (Policy Proxy)	Value-chain movement
🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit App
streamlit run app.py

3️⃣ Open Browser
http://localhost:8501

📁 Project Structure
├── data/
│   ├── import_final.csv
│   ├── export_final.csv
│   ├── production_data.csv
│   ├── gst_state_data.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── forecasting_models.ipynb
│
├── app.py
├── requirements.txt
├── README.md

⚠️ Assumptions & Limitations

GST data is aggregated and used as a proxy (as allowed in problem statement)

Supplier concentration is inferred from dependency ratios

Lithium data is limited to known exploration states

State-wise extraction ≠ processing location (explicitly acknowledged)
