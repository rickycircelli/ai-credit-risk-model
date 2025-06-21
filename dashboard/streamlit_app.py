import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

# Load trained model
import os
model_path = os.path.join(os.path.dirname(__file__), "credit_model.pkl")
model = joblib.load(model_path)

# Page config
st.set_page_config(page_title="Consent-Aware Credit Risk", layout="centered")
st.title("🔐 AI-Powered Credit Risk Estimator")
st.caption("Built with blockchain-style consent toggles")

# Step 1: User Inputs
st.write("### Step 1: Enter Your Info")

years_at_job = st.slider("Years at current job:", 0, 10, 3)
sentiment_score = st.slider("Avg sentiment score (-1 to 1):", -1.0, 1.0, 0.1)
missed_rent = st.slider("Missed rent payments in past 12 months:", 0, 12, 2)

# Dummy values for uncollected features
avg_utility_bill = 150
emoji_usage_rate = 0.2
posts_per_week = 5
rent_on_time_rate = 0.9
job_stability_score = years_at_job / 10
late_payment_flag = 1 if missed_rent > 0 else 0

# Step 2: Consent Toggles
st.write("### Step 2: Choose Data Consent")

consent_employment = st.checkbox("✅ Allow employment data", value=True)
consent_rent = st.checkbox("✅ Allow rent/utilities data", value=True)
consent_social = st.checkbox("✅ Allow social media data", value=True)

# Masking function
def mask(val, allowed):
    return val if allowed else np.nan

# Create full input row
input_data = pd.DataFrame([{
    'years_at_job': mask(years_at_job, consent_employment),
    'job_stability_score': mask(job_stability_score, consent_employment),
    'rent_on_time_rate': mask(rent_on_time_rate, consent_rent),
    'missed_rent_payments': mask(missed_rent, consent_rent),
    'avg_utility_bill': mask(avg_utility_bill, consent_rent),
    'late_payment_flag': mask(late_payment_flag, consent_rent),
    'sentiment_score': mask(sentiment_score, consent_social),
    'emoji_usage_rate': mask(emoji_usage_rate, consent_social),
    'posts_per_week': mask(posts_per_week, consent_social),
    'consent_employment': int(consent_employment),
    'consent_rent': int(consent_rent),
    'consent_social': int(consent_social)
}])

# Split column types
consent_cols = ['consent_employment', 'consent_rent', 'consent_social']
feature_cols = [col for col in input_data.columns if col not in consent_cols]

# Pre-fit imputer on known structure to avoid shape errors
imputer = SimpleImputer(strategy='mean')
imputer.fit(pd.DataFrame([{
    'years_at_job': 5,
    'job_stability_score': 0.5,
    'rent_on_time_rate': 0.9,
    'missed_rent_payments': 1,
    'avg_utility_bill': 150,
    'late_payment_flag': 1,
    'sentiment_score': 0.2,
    'emoji_usage_rate': 0.1,
    'posts_per_week': 5
}]))

# Impute safely, always keeping column count
features_imputed = pd.DataFrame(imputer.transform(input_data[feature_cols]), columns=feature_cols)
input_data_imputed = pd.concat([features_imputed, input_data[consent_cols].reset_index(drop=True)], axis=1)

# Predict
prediction = model.predict_proba(input_data_imputed)[0][1]

# Show output
st.write("### 🔎 Predicted Credit Risk Score")
st.metric(label="Default Risk (0 = low, 1 = high)", value=round(prediction, 3))
