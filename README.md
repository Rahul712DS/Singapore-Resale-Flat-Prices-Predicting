# 🏠 Singapore Resale Flat Price Prediction  

A machine learning project that predicts the resale prices of flats in Singapore based on historical Housing Development Board (HDB) transaction data.  
The project includes **data wrangling, feature engineering, model building, evaluation, and deployment** as an interactive **Streamlit web application**.  

---

## 📌 Problem Statement  
The resale flat market in Singapore is highly competitive, and it can be challenging to estimate the resale value of a flat.  
Many factors affect resale prices, such as:  
- Town  
- Flat type  
- Storey range  
- Floor area  
- Flat model  
- Lease duration  

This project builds a **predictive model** that estimates resale prices and helps buyers/sellers make informed decisions.  

---

## 🚀 Project Workflow  
1. **Data Collection**  
   - Dataset sourced from [Singapore Government Data Portal](https://beta.data.gov.sg/collections/189/view).  
   - Covers resale transactions from 1990 to recent years.  

2. **Data Preprocessing & Feature Engineering**  
   - encoding categorical features, deriving new features:  
     - `age_of_flat = transaction_year - lease_commence_date`  
     - `remaining_lease = 99 - age_of_flat`  

3. **Model Training**  
   - Tried multiple models: Linear Regression, Decision Trees, Random Forest, Gradient Boosted Regressors.  
   - Final choice: **XGBoost Regressor** (best performance).  

4. **Model Evaluation**  
   - Mean Absolute Error (MAE): ~16,631  
   - Root Mean Squared Error (RMSE): ~25,107  
   - R² Score: ~0.98  

5. **Deployment**  
   - Built with **Streamlit**.  

---

## 📊 Model Performance  
| Metric | Value |
|--------|-------|
| MAE    | 16,631 |
| RMSE   | 25,107 |
| R²     | 0.98 |

✅ The model achieves strong predictive power with an average error of ~7–10% of resale price.  

---

## 🌐 Streamlit Web Application  
The web app allows users to:  
- Select flat details (year, month, town, flat type, storey, floor area, model, age).  
- Automatically compute **remaining lease**.  
- View **historical average resale prices** for selected filters.  
- Get a **predicted resale price with confidence interval**.  

### 🔎 Example Output
<img width="2677" height="1350" alt="image" src="https://github.com/user-attachments/assets/778e8bf1-f58e-4494-8788-4564db09d0b6" />

### 🧑‍💻 Comparing Predicted Result of Model With Actual Data Set Values

### Model Prediction
<img width="905" height="266" alt="image" src="https://github.com/user-attachments/assets/dbb8da44-85c9-44a5-84a4-7011a50a15a1" />

### Actual Dataset
<img width="2731" height="454" alt="image" src="https://github.com/user-attachments/assets/ceafe72d-f26f-48d6-aba7-8e71bd8b035f" />

### we can observe that there is 1.79% of error.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Libraries**: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit, joblib  
- **Deployment**: Streamlit 

---

## 📈 Future Improvements
- Add more advanced feature engineering (price per sqm, town × flat type interactions).  
- Incorporate macroeconomic factors (CPI, interest rates).  
- Deploy using **Docker** for portability.  
- Add API endpoint for external access.  

---

## 📹 Demo Video  
🔗 [LinkedIn Post / YouTube Link – *Add your demo video link here*]  

---

## 👨‍💻 Author  
**Rahul Raj**  
- Aspiring Data Scientist | Data Analyst 
- [LinkedIn Profile](https://www.linkedin.com/in/rahul-raj-22534a14b/)  
- [GitHub Portfolio](https://github.com/Rahul712DS)  

 
