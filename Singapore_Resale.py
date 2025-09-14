import pandas as pd
from pathlib import Path
import json
import joblib

def preprocess_and_save(input_files, output_file):
    # Step 1: Concatenate all data
    dfs = [pd.read_csv(file) for file in input_files]
    df = pd.concat(dfs, ignore_index=True)

    # Convert month column
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    # Split storey and create midpoint
    df[["storey_lower", "storey_upper"]] = df["storey_range"].str.split(" TO ", expand=True).astype(int)
    df["storey_mid"] = (df["storey_lower"] + df["storey_upper"]) / 2

    # Step 4: Label encoding using Factorize + mapping export
    def encode_and_save(col, filename):
        codes, uniques = pd.factorize(df[col])
        df[f"{col}_code"] = codes
        mapping = {str(val): idx for idx, val in enumerate(uniques)}
        with open(output_file / filename, "w") as f:
            json.dump(mapping, f)
        return df[f"{col}_code"]
    
    df["town_code"] = encode_and_save("town", "town_mapping.json")
    df["flat_type_code"] = encode_and_save("flat_type", "flat_type_mapping.json")
    df["flat_model_code"] = encode_and_save("flat_model", "flat_model_mapping.json")

    # Feature engineering
    df["age_of_flat"] = abs(df["year"] - df["lease_commence_date"])

    # ✅ Always recompute remaining_lease (ignore existing column to avoid NaN issue)
    df["remaining_lease"] = 99 - df["age_of_flat"]

    # Build cleaned dataset
    df_cleaned = pd.DataFrame({
        "year": df["year"],
        "month_num": df["month_num"],
        "flat_type_code": df["flat_type_code"],
        "storey_lower": df["storey_lower"],
        "storey_mid": df["storey_mid"],
        "storey_upper": df["storey_upper"],
        "floor_area_sqm": df["floor_area_sqm"],
        "town_code": df["town_code"],
        "flat_model_code": df["flat_model_code"],
        "age_of_flat": df["age_of_flat"],
        "remaining_lease": df["remaining_lease"],
        "resale_price": df["resale_price"]
    })

    # Save cleaned CSV
    df_cleaned.to_csv(output_file / "cleaned_all_data.csv", index=False)
    print(f"✅ Saved cleaned data to {output_file}")



# -------- Run on multiple files --------
input_files = [
    r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices\Resale Flat Prices From 1990 - 1999.csv",
    r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices\Resale Flat Prices From 2000 - Feb 2012.csv",
    r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices\Resale Flat Prices From Mar 2012 to Dec 2014.csv",
    r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices\Resale Flat Prices From Jan 2015 to Dec 2016.csv",
    r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices\Resale Flat prices From Jan-2017 onwards.csv",
]

output_file = Path(r"C:\Users\rahul\Data Science\DATASET\Singapore Resale Flat Prices Predicting Data Set\ResaleFlatPrices")
preprocess_and_save(input_files, output_file)

# Load cleaned data
df = pd.read_csv(output_file/ "cleaned_all_data.csv")

#feature and target column selection
X = df.drop(columns=["resale_price"])
y = df["resale_price"]
print(X.head())
print(y.head()) 

#Model training and evaluation can be done hereafter

#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Training set head:", X_train.head())
print("Test set head:", X_test.head())

# Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
model_linear = LinearRegression()

model_linear.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)
print(y_pred_linear[:5])

print("Linear Regression Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_linear))   
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print(f'root_mean_squared_error:\n',np.sqrt(mean_squared_error(y_test, y_pred_linear)))
print("R^2 Score:", r2_score(y_test, y_pred_linear))

#check the price with some random values
sample = pd.DataFrame([{
    "year": 1990,
    "month_num": 1,
    "flat_type_code": 1,
    "storey_lower": 1,
    "storey_mid": 5,
    "storey_upper": 10,
    "floor_area_sqm": 73,
    "town_code": 0,
    "flat_model_code": 1,
    "age_of_flat": 14,
    "remaining_lease": 85
}])


y_pred_linear = model_linear.predict(sample)
print("Predicted resale price:", y_pred_linear[0])

# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model_dtr = DecisionTreeRegressor(random_state=0, max_depth=20)

model_dtr.fit(X_train, y_train)
y_pred_dtr = model_dtr.predict(X_test)
print(y_pred_dtr[:5])

print("Decision Tree Regressor Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_dtr))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_dtr))
print(f'root_mean_squared_error:\n',np.sqrt(mean_squared_error(y_test, y_pred_dtr)))
print("R^2 Score:", r2_score(y_test, y_pred_dtr))

y_pred_dtr = model_linear.predict(sample)
print("Predicted resale price:", y_pred_dtr[0])

# Grid Search for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    'max_depth': [10, 20, 30]
}
grid_search = GridSearchCV(estimator=model_dtr, param_grid=param_grid)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
model_rfr = RandomForestRegressor(random_state=0, n_estimators=10, max_depth=20)
model_rfr.fit(X_train, y_train)
y_pred_rfr = model_rfr.predict(X_test)
print(y_pred_rfr[:5])

print("Random Forest Regressor Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rfr))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rfr))
print(f'root_mean_squared_error:\n',np.sqrt(mean_squared_error(y_test, y_pred_rfr)))
print("R^2 Score:", r2_score(y_test, y_pred_rfr))

#grid search for random forest
param_grid = {
    'n_estimators': [10, 20, 30,40,50],  
    'max_depth': [10, 20, 30,40,50],
    
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=model_rfr, param_grid=param_grid)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   
model_gbr = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=5)
#n_estimators=100 mean 100 trees will be built
#learning_rate=0.1 means each tree will contribute 10% to the final prediction
#max_depth=3 means each tree can have a maximum depth of 3
model_gbr.fit(X_train, y_train)
y_pred_gbr = model_gbr.predict(X_test)
print(y_pred_gbr[:5])

print("Gradient Boosting Regressor Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_gbr))      
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_gbr))
print(f'root_mean_squared_error:\n',np.sqrt(mean_squared_error(y_test, y_pred_gbr)))
print("R^2 Score:", r2_score(y_test, y_pred_gbr))

# Gradient XGBoost Regressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  

model_xgbr = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=350, learning_rate=0.1, max_depth=15)
model_xgbr.fit(X_train, y_train)


y_pred_xgbr = model_xgbr.predict(X_test)
print(y_pred_xgbr[:5])

# Metrics
mae = mean_absolute_error(y_test, y_pred_xgbr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgbr))
r2 = model_xgbr.score(X_test, y_test)

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

#Model dumping
joblib.dump(model_xgbr, r"C:\Users\rahul\OneDrive\Desktop\VS Code\Singapore Resale Flat\xgboost_model.pkl")
print("✅Model saved as xgboost_model.pkl")

with open(r"C:\Users\rahul\OneDrive\Desktop\VS Code\Singapore Resale Flat\metrics.json", "w") as f:
    json.dump(metrics, f)

print("XGBoost Regressor Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_xgbr)) 
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_xgbr))
print(f'root_mean_squared_error:\n',np.sqrt(mean_squared_error(y_test, y_pred_xgbr)))
print("R^2 Score:", r2_score(y_test, y_pred_xgbr))


