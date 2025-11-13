import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os
import plotly.express as px
st.set_page_config(page_title="🌾 Crop Yield Predictor & Insights", layout="wide", page_icon="🌱")
try:
    MODEL_PATH = "Projects/best_model_compressed.joblib"
    SCALER_PATH = "Projects/scaler.joblib"
    LE_CROP_PATH = "Projects/le_crop.joblib"
    LE_STATE_PATH = "Projects/le_state.joblib"
    LE_SEASON_PATH = "Projects/le_season.joblib"
    CSV_PATH = "Projects/merged_crop_yield_dataset.csv"

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_crop = joblib.load(LE_CROP_PATH)
    le_state = joblib.load(LE_STATE_PATH)
    le_season = joblib.load(LE_SEASON_PATH)
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Error loading model/scaler/encoders/csv: {e}")
    st.stop()

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_path = "Projects/photo-1623190632241-20a391a7b2e0.jpg"
base64_image = get_base64_of_bin_file(bg_image_path) if os.path.exists(bg_image_path) else ""

page_css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
div.block-container {{
    background-color: rgba(0, 0, 0, 0.6);
    border-radius: 20px;
    padding: 3rem;
    color: #fff;
    box-shadow: 0 0 25px rgba(0,0,0,0.5);
    max-width: 1100px;
    margin: auto;
    backdrop-filter: blur(6px);
}}
h1, h2, h3, h4, h5, h6, label, p {{
    color: #fff !important;
}}
.stButton > button {{
    background: linear-gradient(90deg, #00c853, #009624);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}}
.stButton > button:hover {{
    transform: scale(1.03);
}}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)
sidebar_glass_style = """
<style>
[data-testid="stSidebar"] {
    background: rgba(25, 25, 35, 0.6); /* semi-transparent dark */
    backdrop-filter: blur(18px) saturate(180%);
    -webkit-backdrop-filter: blur(18px) saturate(180%);
    border-right: 1.5px solid rgba(255, 255, 255, 0.2);
    box-shadow: 4px 0 25px rgba(0, 0, 0, 0.6);
    border-top-right-radius: 25px;
    border-bottom-right-radius: 25px;
    color: white;
    transition: all 0.4s ease-in-out;
}

/* Subtle inner glow */
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 25px;
    background: radial-gradient(circle at top left, rgba(255,255,255,0.15), transparent 60%);
    pointer-events: none;
}

/* Sidebar hover depth */
[data-testid="stSidebar"]:hover {
    background: rgba(30, 30, 45, 0.7);
    backdrop-filter: blur(22px) saturate(200%);
    -webkit-backdrop-filter: blur(22px) saturate(200%);
    transform: translateX(2px);
}

/* Sidebar text styling */
[data-testid="stSidebar"] * {
    color: #f2f2f2 !important;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar title */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    font-weight: 700;
    color: #e8ffe8 !important;
}

/* Sidebar radio buttons */
.stRadio > div {
    background: rgba(255, 255, 255, 0.05);
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
"""
dark_expander_css = """
<style>
/* Dark header for expander (both collapsed & expanded) */
div[data-testid="stExpander"] > div:first-child {
    background-color: rgba(15, 15, 15, 0.85) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 15px !important;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 0 10px rgba(0,0,0,0.6);
    transition: background-color 0.3s ease-in-out;
}

/* Slight hover effect */
div[data-testid="stExpander"] > div:first-child:hover {
    background-color: rgba(25, 25, 25, 0.95) !important;
}

/* Inside content area */
div[data-testid="stExpander"] div[role="region"] {
    background-color: rgba(10, 10, 10, 0.75) !important;
    border-radius: 10px !important;
    padding: 15px !important;
}
</style>
"""
st.markdown(dark_expander_css, unsafe_allow_html=True)

st.markdown(dark_expander_css, unsafe_allow_html=True)

st.markdown(sidebar_glass_style, unsafe_allow_html=True)

st.sidebar.title("🌿 Navigation")
page = st.sidebar.radio("Go to:", ["🔍 Yield Prediction", "📊 Insights Dashboard"])

if page == "🔍 Yield Prediction":
    st.title("🌾 Crop Yield Predictor")
    st.markdown("<p style='color:#e6ffe6'>Predict crop yield (tons/hectare) using a trained Random Forest model.Get accurate insights on productivity based on soil nutrients, climate, rainfall, and seasonal factors.Visualize top-performing states and crops, and explore data-driven recommendations to improve yield.</p>", unsafe_allow_html=True)

    # Dropdown options
    crop_options = sorted(df['Crop'].dropna().unique().tolist())
    state_options = sorted(df['State'].dropna().unique().tolist())
    season_options = sorted(df['Season'].dropna().unique().tolist())

    # Input Form
    st.subheader("🧮 Enter Input Details")
    col_left, col_right = st.columns([0.9, 0.9])
    with col_left:
        crop = st.selectbox("🌱 Crop", ["Select Crop"] + crop_options)
        state = st.selectbox("🏞️ State", ["Select State"] + state_options)
        season = st.selectbox("☀️ Season", ["Select Season"] + season_options)
    with col_right:
        area = st.number_input("🌾 Area (hectares)", value=0.0, min_value=0.0, step=0.01)
        production = st.number_input("🏭 Production (tons)", value=0.0, min_value=0.0, step=0.01)
        rainfall = st.number_input("🌧️ Annual Rainfall (mm)", value=0.0, min_value=0.0, step=0.1)

    with st.expander("⚙️ Advanced Inputs"):
        a1, a2, a3 = st.columns(3)
        with a1:
            fertilizer = st.number_input("🧴 Fertilizer (kg)", value=0.0, min_value=0.0, step=1.0)
        with a2:
            n = st.number_input("🌿 Nitrogen (N)", value=0.0, min_value=0.0, step=0.1)
        with a3:
            k = st.number_input("🧂 Potassium (K)", value=0.0, min_value=0.0, step=0.1)
        b1, b2, b3 = st.columns(3)
        with b1:
            pesticide = st.number_input("🧫 Pesticide (kg)", value=0.0, min_value=0.0, step=0.1)
        with b2:
            p = st.number_input("🧪 Phosphorus (P)", value=0.0, min_value=0.0, step=0.1)
        with b3:
            temperature = st.number_input("🌡️ Temperature (°C)", value=0.0, min_value=-10.0, max_value=60.0, step=0.1)
        c1, c2 = st.columns(2)
        with c1:
            humidity = st.number_input("💧 Humidity (%)", value=0.0, min_value=0.0, max_value=100.0, step=0.1)
        with c2:
            crop_year = st.number_input("📅 Crop Year", value=2024, min_value=1980, max_value=2035, step=1)

    if st.button("🔍 Predict Yield", use_container_width=True):
        if crop == "Select Crop" or state == "Select State" or season == "Select Season":
            st.warning("⚠️ Please select valid Crop, State, and Season.")
        else:
            try:
                crop_enc = le_crop.transform([crop])[0]
                state_enc = le_state.transform([state])[0]
                season_enc = le_season.transform([season])[0]

                input_data = np.array([[crop_enc, crop_year, season_enc, state_enc,
                                        area, production, rainfall, fertilizer, pesticide,
                                        n, p, k, temperature, humidity]])
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                pred = np.clip(pred, 0, 10)

                st.success(f"🌾 **Predicted Yield:** {pred:.2f} tons/hectare")

                # Top 5 States/Crops Graph
                st.markdown("### 📈 Insights from Dataset")
                col1, col2 = st.columns(2)

                with col1:
                    crop_df = df[df['Crop'] == crop]
                    if not crop_df.empty:
                        top_states = crop_df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(5).reset_index()
                        fig1 = px.bar(top_states, x='State', y='Yield', color='Yield',
                                      color_continuous_scale='Greens',
                                      title=f"🏆 Top 5 States for {crop} (Avg Yield)")
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("No data for selected crop.")

                with col2:
                    state_df = df[df['State'] == state]
                    if not state_df.empty:
                        top_crops = state_df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(5).reset_index()
                        fig2 = px.bar(top_crops, x='Crop', y='Yield', color='Yield',
                                      color_continuous_scale='Blues',
                                      title=f"🌾 Top 5 Crops in {state} (Avg Yield)")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No data for selected state.")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

elif page == "📊 Insights Dashboard":
    st.title("📊 Crop Yield Insights Dashboard")
    st.markdown("Explore dataset patterns, feature importance, and yield statistics.")

    # Summary
    st.subheader("📋 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Total Records", len(df))
    c2.metric("🌾 Unique Crops", df['Crop'].nunique())
    c3.metric("🏞️ Unique States", df['State'].nunique())

    # Feature Importance
    st.subheader("🧠 Feature Importance (from Model)")
    try:
        importances = model.feature_importances_
        features = ['Crop','Crop_Year','Season','State','Area','Production',
                    'Annual_Rainfall','Fertilizer','Pesticide','N','P','K','Temperature','Humidity']
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig_imp = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='viridis',
                         title='🔍 Top 10 Influential Features')
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

    # Yield Distribution
    st.subheader("🌾 Yield Distribution")
    fig_yield = px.histogram(df, x='Yield', nbins=40, color_discrete_sequence=['#00c853'],
                             title='Yield Distribution (tons/hectare)')
    st.plotly_chart(fig_yield, use_container_width=True)

    # Top performers
    st.subheader("🏆 Top Performers")
    col1, col2 = st.columns(2)
    with col1:
        top_crops = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_crop = px.bar(top_crops, x='Yield', y='Crop', orientation='h', color='Yield',
                          color_continuous_scale='Blues', title='🌿 Top 10 Crops by Avg Yield')
        st.plotly_chart(fig_crop, use_container_width=True)
    with col2:
        top_states = df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_state = px.bar(top_states, x='Yield', y='State', orientation='h', color='Yield',
                           color_continuous_scale='Greens', title='🏞️ Top 10 States by Avg Yield')
        st.plotly_chart(fig_state, use_container_width=True)

    # Text Summary
    st.subheader("🪄 Insights Summary")
    avg_yield = df['Yield'].mean()
    best_crop = top_crops.iloc[0]['Crop']
    best_state = top_states.iloc[0]['State']
    st.markdown(f"""
    ✅ The average yield is **{avg_yield:.2f} tons/hectare**.  
    🥇 {best_crop} has the highest average yield among crops.  
    🏆 {best_state} leads among all states in yield performance.  
    🔬 Features like **Production**, **Area**, and **Rainfall** have the most influence on yield.
    """)
