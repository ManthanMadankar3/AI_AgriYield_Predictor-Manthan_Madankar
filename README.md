🌾 Crop Yield Prediction Using Machine Learning

Harnessing machine learning to deliver accurate crop yield predictions and support data-driven agricultural planning.

📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict crop yield using environmental, soil, and agricultural parameters.
The system includes:

Data Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Machine Learning Model Training (Random Forest)

Model Evaluation

Model Saving & Compression

Streamlit-based interactive web application

🗂 Dataset

The dataset contains crop-related environmental and production attributes such as:

Crop

State

Season

Area

Production

Annual Rainfall

N, P, K (Nutrient values)

Fertilizer

Pesticide

Temperature

Humidity

Crop Year

Yield (Target variable)

🔧 Technologies & Libraries Used
📊 Data Processing

Pandas – Data cleaning & manipulation

NumPy – Numerical computations

📈 Visualization

Matplotlib

Seaborn

Plotly (for interactive charts in Streamlit)

🤖 Machine Learning

Scikit-Learn

LabelEncoder

StandardScaler

Train-Test Split

RandomForestRegressor

Evaluation Metrics (R², RMSE, MAE)

💾 Model Persistence

Joblib – Saving model, encoders, and scaler

LZMA Compression – Creating optimized compressed model file

🌐 Deployment

Streamlit – Web application for prediction & dashboard

🔍 Model Training Summary

Algorithm: Random Forest Regressor

R² Score: 0.976

RMSE: 0.259

MAE: 0.114

Handles non-linear agricultural data with high accuracy

Robust to outliers and noisy real-world data

🚀 Features of the Web Application
🔮 Yield Prediction

Enter values such as crop, state, season, rainfall, area, NPK nutrients, temperature, humidity, fertilizer, and pesticide
→ Receive predicted crop yield instantly.

📊 Interactive Insights Dashboard

Visualizes:

Top performing crops

State-wise yield patterns

Yield distribution

Feature importance

Dataset overview (records, crops, states)

🧹 Preprocessing Steps

Handling missing values

Removing invalid values (area or yield ≤ 0)

Outlier filtering (Yield < 10)

Label Encoding categorical variables

Standard scaling numerical values

Splitting dataset into training & testing sets

🧠 Machine Learning Pipeline

Load & clean dataset

Perform EDA

Encode categorical variables

Scale numerical features

Split dataset

Train model (Random Forest)

Evaluate performance

Save & compress model

Deploy with Streamlit

🖥 How to Run the Application
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Streamlit App
streamlit run app.py

3️⃣ Use the Web Interface

Enter crop details

View predicted yield

Explore insights dashboard

📦 Project Structure
├── merged_crop_yield_dataset.csv
├── best_model.joblib
├── best_model_compressed.joblib
├── scaler.joblib
├── le_crop.joblib
├── le_state.joblib
├── le_season.joblib
├── app.py
├── model_training.py
├── README.md
└── images/

🌱 Future Enhancements

Integrate real-time weather API

Add NDVI & satellite data

Use Deep Learning (LSTM / DNN)

Deploy as a cloud-based API

Mobile-friendly interface
