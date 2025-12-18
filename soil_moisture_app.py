import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# 🌱 App Configuration
# ===============================
st.set_page_config(page_title="Smart Soil Moisture App", layout="wide")
st.title("🌱 Smart Soil Moisture Model")

st.markdown("""
This model predicts **soil moisture** using temperature, humidity, rainfall, soil pH, crop type, and fertilizer type.  
Use the sidebar to tweak ML parameters and see how accuracy changes.
""")

# ===============================
# 🔧 Sidebar Tuning Options
# ===============================
st.sidebar.header("🔧 Model Tuning")
split_ratio = st.sidebar.slider("Train/Test Split (Test %)", 10, 50, 20, step=5)
tree_depth = st.sidebar.slider("🌳 Decision Tree Depth", 2, 15, 5)
nn_layer_size = st.sidebar.slider("🤖 Neural Net Layer Size", 10, 200, 50, step=10)

# ===============================
# 1. Load Data
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Expanded_Mixup_LatLong.csv")

df = load_data()

# ===============================
# 2. Filters
# ===============================
col0, col1, col2, col3 = st.columns(4)
with col0:
    region = st.selectbox("Region", df["region"].unique())
with col1:
    crop = st.selectbox("Crop", df["crop_type"].unique())
with col2:
    fertilizer = st.selectbox("Fertilizer", df["fertilizer_type"].unique())
with col3:
    feature_x = st.selectbox("X-Axis Feature", ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"])

filtered_df = df[
    (df["region"] == region) &
    (df["crop_type"] == crop) &
    (df["fertilizer_type"] == fertilizer)
].copy()

# ===============================
# 3. Moisture Detection + Binning
# ===============================
if "soil_moisture_%" in df.columns:
    soil_col = "soil_moisture_%"
elif "soil_moisture" in df.columns:
    soil_col = "soil_moisture"
else:
    candidates = [c for c in df.columns if "moisture" in c.lower()]
    soil_col = candidates[0] if candidates else st.stop()

st.caption(f"Using soil column: **{soil_col}**")

bins = [0, 30, 60, 100]
labels = ["Dry", "Optimal", "Wet"]
filtered_df["Soil_Moisture_Level"] = pd.cut(filtered_df[soil_col], bins=bins, labels=labels)

# ===============================
# 4. Visualize Data
# ===============================
if not filtered_df.empty:
    fig = px.scatter(
        filtered_df,
        x=feature_x,
        y=soil_col,
        color="Soil_Moisture_Level",
        title=f"Soil Moisture vs {feature_x}",
        labels={soil_col: "Soil Moisture (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 5. ML Feature Engineering
# ===============================
features = ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]
X = filtered_df[features].copy()
y = filtered_df[soil_col]

le_crop = LabelEncoder()
le_fert = LabelEncoder()
X["crop_type_encoded"] = le_crop.fit_transform(filtered_df["crop_type"])
X["fertilizer_type_encoded"] = le_fert.fit_transform(filtered_df["fertilizer_type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio / 100, random_state=42)

# Scale for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 6. Train ML Models
# ===============================
# Decision Tree
dt_model = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Neural Net
nn_model = MLPRegressor(hidden_layer_sizes=(nn_layer_size, nn_layer_size), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)

# ===============================
# 7. Model Evaluation Metrics
# ===============================
st.subheader("📐 Model Performance Metrics")

# Decision Tree Metrics
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_std = np.std(abs(y_test - dt_pred))

# Neural Net Metrics
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_mse = mean_squared_error(y_test, nn_pred)
nn_rmse = np.sqrt(nn_mse)
nn_std = np.std(abs(y_test - nn_pred))

colA, colB, colC, colD = st.columns(4)
colA.metric("🌳 DT MAE", f"{dt_mae:.2f}")
colB.metric("🌳 DT RMSE", f"{dt_rmse:.2f}")
colC.metric("🌳 DT MSE", f"{dt_mse:.2f}")
colD.metric("🌳 DT Std Dev", f"{dt_std:.2f}")

colA2, colB2, colC2, colD2 = st.columns(4)
colA2.metric("🤖 NN MAE", f"{nn_mae:.2f}")
colB2.metric("🤖 NN RMSE", f"{nn_rmse:.2f}")
colC2.metric("🤖 NN MSE", f"{nn_mse:.2f}")
colD2.metric("🤖 NN Std Dev", f"{nn_std:.2f}")

# ===============================
# 8. Latest Prediction Summary
# ===============================
latest_input = X.tail(1)
dt_latest = dt_model.predict(latest_input)[0]
nn_latest = nn_model.predict(scaler.transform(latest_input))[0]
avg = (dt_latest + nn_latest) / 2

st.subheader("💧 Latest Moisture Prediction")
col1, col2 = st.columns(2)
col1.markdown(f"<h3 style='color:#2DBBCC;'>🌳 DT: {dt_latest:.2f}%</h3>", unsafe_allow_html=True)
col2.markdown(f"<h3 style='color:#2DBBCC;'>🤖 NN: {nn_latest:.2f}%</h3>", unsafe_allow_html=True)

# Moisture condition bar
if avg < 30:
    condition = "🌵 Dry — Needs Water"
    color = "red"
elif avg < 60:
    condition = "🌾 Optimal"
    color = "green"
else:
    condition = "💧 Too Wet"
    color = "blue"

st.progress(int(avg))
st.markdown(f"<p style='color:{color}; font-size:18px;'>{condition}</p>", unsafe_allow_html=True)
