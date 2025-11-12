import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# ===============================
# 🌱 App Configuration
# ===============================
st.set_page_config(page_title="Smart Soil Moisture App", layout="wide")
st.title("🌱 Smart Soil Moisture Model")

st.markdown("""
This model allows you to **explore soil moisture** levels depending on temperature, humidity, rainfall, and other variables.
""")

# ===============================
# 1. Load Data
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")

df = load_data()

# ===============================
# 2. Filters
# ===============================
col1, col2, col3 = st.columns(3)
with col1:
    crop = st.selectbox("Select Crop Type:", options=df["crop_type"].unique())
with col2:
    fertilizer = st.selectbox("Select Fertilizer:", options=df["fertilizer_type"].unique())
with col3:
    feature_x = st.selectbox("Select X-Axis Feature:", options=["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"])

filtered_df = df[(df["crop_type"] == crop) & (df["fertilizer_type"] == fertilizer)].copy()

# ===============================
# 3. Soil Moisture Classification
# ===============================
bins = [0, 30, 60, 100]
labels = ["Dry", "Optimal", "Wet"]
soil_col = [col for col in df.columns if "moisture" in col.lower()][0]

filtered_df.loc[:, "Soil_Moisture_Level"] = pd.cut(filtered_df[soil_col], bins=bins, labels=labels)

# ===============================
# 4. Model Preparation
# ===============================
X = filtered_df[[feature_x]].copy()

# Encode categorical features
le_crop = LabelEncoder()
le_fert = LabelEncoder()
X["crop_type_encoded"] = le_crop.fit_transform(filtered_df["crop_type"])
X["fertilizer_type_encoded"] = le_fert.fit_transform(filtered_df["fertilizer_type"])

y = filtered_df["soil_moisture_%"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Train Models
# ===============================
# 🌳 Decision Tree Model
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

# 🤖 Neural Network Model
nn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))

# ===============================
# 6. Actual vs Predicted Comparison
# ===============================
st.subheader("📉 Model Accuracy Comparison: Actual vs Predicted Soil Moisture")

col1, col2 = st.columns(2)

# --- Decision Tree ---
with col1:
    st.markdown("### 🌳 Decision Tree Results")

    dt_compare_df = pd.DataFrame({
        "Actual Soil Moisture (%)": y_test.values,
        "Predicted Soil Moisture (%)": dt_pred
    }).reset_index(drop=True)
    dt_compare_df["Error (%)"] = abs(dt_compare_df["Actual Soil Moisture (%)"] - dt_compare_df["Predicted Soil Moisture (%)"])

    st.dataframe(dt_compare_df, use_container_width=True)

    fig_dt = px.scatter(
        dt_compare_df,
        x="Actual Soil Moisture (%)",
        y="Predicted Soil Moisture (%)",
        title="Decision Tree: Actual vs Predicted Soil Moisture",
        color="Error (%)",
        color_continuous_scale="Viridis",
        trendline="ols"
    )
    fig_dt.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red", dash="dash")
    )
    fig_dt.update_layout(height=400)
    st.plotly_chart(fig_dt, use_container_width=True)

# --- Neural Network ---
with col2:
    st.markdown("### 🤖 Neural Network Results")

    nn_compare_df = pd.DataFrame({
        "Actual Soil Moisture (%)": y_test.values,
        "Predicted Soil Moisture (%)": nn_pred
    }).reset_index(drop=True)
    nn_compare_df["Error (%)"] = abs(nn_compare_df["Actual Soil Moisture (%)"] - nn_compare_df["Predicted Soil Moisture (%)"])

    st.dataframe(nn_compare_df, use_container_width=True)

    fig_nn = px.scatter(
        nn_compare_df,
        x="Actual Soil Moisture (%)",
        y="Predicted Soil Moisture (%)",
        title="Neural Network: Actual vs Predicted Soil Moisture",
        color="Error (%)",
        color_continuous_scale="Viridis",
        trendline="ols"
    )
    fig_nn.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red", dash="dash")
    )
    fig_nn.update_layout(height=400)
    st.plotly_chart(fig_nn, use_container_width=True)

# ===============================
# 7. Model Performance Summary
# ===============================
st.markdown("---")
st.subheader("📊 Model Performance Summary")

col3, col4 = st.columns(2)
with col3:
    st.metric("🌳 Decision Tree RMSE", f"{dt_rmse:.2f}%")
with col4:
    st.metric("🤖 Neural Network RMSE", f"{nn_rmse:.2f}%")

if nn_rmse < dt_rmse:
    st.success("✅ The Neural Network provides more accurate soil moisture predictions.")
else:
    st.warning("⚠️ The Decision Tree performed slightly better in this case.")

# ===============================
# 8. Latest Predictions and Interpretation
# ===============================
latest_features = X.tail(1)
pred_dt_latest = dt_model.predict(latest_features)[0]
pred_nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

st.markdown("---")
st.markdown("### 💧 Latest Soil Moisture Predictions")

col5, col6 = st.columns(2)
with col5:
    st.markdown(f"**🌳 Decision Tree Prediction:** <h2 style='color:#2DBBCC;'>{pred_dt_latest:.2f}%</h2>", unsafe_allow_html=True)
with col6:
    st.markdown(f"**🤖 Neural Network Prediction:** <h2 style='color:#2DBBCC;'>{pred_nn_latest:.2f}%</h2>", unsafe_allow_html=True)

predicted_value = (pred_nn_latest + pred_dt_latest) / 2
if predicted_value < 30:
    condition = "🌵 **Dry Soil** – Irrigation Needed"
    bar_color = "red"
elif predicted_value < 60:
    condition = "🌾 **Optimal Moisture** – Ideal Conditions"
    bar_color = "green"
else:
    condition = "💧 **Wet Soil** – Overwatered"
    bar_color = "blue"

st.markdown("### 🌡️ Soil Moisture Condition")
st.progress(int(predicted_value))
st.markdown(f"<p style='color:{bar_color}; font-size:18px;'>{condition}</p>", unsafe_allow_html=True)

st.info("""
**ℹ️ What is RMSE?**  
RMSE (Root Mean Squared Error) measures how close the model’s predictions are to the actual soil moisture values.  
Lower RMSE → More accurate predictions.  

**Interpretation:**  
- Below 30% → Soil is dry and may need watering.  
- 30–60% → Moisture is in the optimal range.  
- Above 60% → Soil is too wet; reduce irrigation.
""")
