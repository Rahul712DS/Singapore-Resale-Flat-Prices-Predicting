import streamlit as st
import pandas as pd
from pathlib import Path     
import numpy as np
import json
import joblib


#Loading path of Clean Dataset
loc = Path(r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices")

# Load mapping
with open("town_mapping.json", "r") as f:
    town_mapping = json.load(f)

with open("flat_type_mapping.json", "r") as f:
    flat_type_mapping = json.load(f)

with open("flat_model_mapping.json", "r") as f:
    flat_model_mapping = json.load(f)
    
st.image(r"C:\Users\rahul\OneDrive\Desktop\VS Code\Singapore Resale Flat\Marina.jpg", width=200)
st.title("🏠 Singapore Resale Flat Price Prediction", width="stretch")

st.set_page_config(page_title="🏠 Singapore Resale Flat Price Prediction", layout="wide")

# Load model
model = joblib.load(r"C:\Users\rahul\OneDrive\Desktop\VS Code\Singapore Resale Flat\xgboost_model.pkl")

#load Clean data
df = pd.read_csv(loc/"cleaned_all_data.csv")

# Show text options to user
year_range = (i+1  for i in range(1989,2027))
month_range = (i+1 for i in range(0,12))
#st.write(year_range)
col1, col2, col3  = st.columns(3)
with col1:
    year = st.slider("Select Year",min_value=1990, max_value=2030, value=1990)
    month = st.selectbox("Select Month",month_range)
    flat_type = st.selectbox("Select Flat Type", list(flat_type_mapping.keys()))

with col2:
    storey_lower = st.slider("Select Storey Lower",min_value=1, max_value=50, value=1)
    storey_upper = st.slider("Select Storey Upper", min_value=1, max_value=51, value=2)
    floor_area_sqm = st.slider("Select Floor Area sqm", min_value=10, max_value=300, value=50)
    if storey_upper < storey_lower:
        st.error("Error: Storey Upper must be greater than or equal to Storey Lower")
    else:
        storey_mid = storey_lower + storey_upper / 2
        with col3:
            
            town = st.selectbox("Select Town", list(town_mapping.keys()))
            flat_model = st.selectbox("Select Flat Model", list(flat_model_mapping.keys()))
            age_of_flat = st.slider("Select Age of Flat", min_value=0, max_value=99, value=10)
            remaining_lease = 99 - age_of_flat
            st.write(f"Remaining Lease (automatically computed): {99 - age_of_flat} years")

    
        # Convert text → code
        town_code = town_mapping[town]
        flat_type_code = flat_type_mapping[flat_type]
        flat_model_code = flat_model_mapping[flat_model]

        # Prepare features
        features = pd.DataFrame([{
            "year" : year,
            "month_num": month,
            "flat_type_code": flat_type_mapping[flat_type],
            "storey_lower" : storey_lower,
            "storey_mid"   : storey_mid,
            "storey_upper" : storey_upper,
            "floor_area_sqm" : floor_area_sqm,
            "town_code" : town_mapping[town],
            "flat_model_code" : flat_model_mapping[flat_model],
            "age_of_flat" : age_of_flat,
            "remaining_lease" : remaining_lease
        }])
        
        # Average price per year for selected filters
        filtered_df = df[(df["year"] == year) & 
                         (df["floor_area_sqm"] == floor_area_sqm) &
                         (df["age_of_flat"] == age_of_flat)
                         ]
        
        avg = filtered_df.groupby(["year", "floor_area_sqm", "age_of_flat"])["resale_price"].mean().reset_index()
        if avg.empty:
            st.warning("No historical data available for the selected inputs.")    
        else:
            st.write("### Average Resale Prices for Selected Inputs")
            st.table(avg.reset_index(drop=True))
       
        # Predict button
        if st.button("Predict Price"):
            prediction = model.predict(features)[0]
            with open("metrics.json", "r") as f:
                 metrics = json.load(f)
            
            RMSE = metrics["RMSE"]
            lower_bound = prediction - RMSE
            upper_bound = prediction + RMSE
            st.success(f"Estimated Resale Price: SGD {prediction:,.0f}")
            st.info(f"Confidence Interval: SGD {lower_bound:,.0f} – SGD {upper_bound:,.0f}")    
