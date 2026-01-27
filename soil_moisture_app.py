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
This model predicts **soil moisture** based on temperature, humidity, rainfall, soil pH, crop, and fertilizer type.
""")


# ===============================
# ⚙️ Model Tuning Controls
# ===============================
st.sidebar.header("⚙️ Model Tuning Controls")

test_size = st.sidebar.slider("Train/Test Split (Test Size)", 0.1, 0.4, 0.2, 0.05)
tree_depth = st.sidebar.slider("Decision Tree Max Depth", 2, 15, 5)
nn_layer_size = st.sidebar.slider("Neural Network Hidden Layer Size", 16, 128, 50, 16)


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
    region = st.selectbox("Select Region:", df["region"].unique())

df["location_label"] = df["latitude"].round(4).astype(str) + ", " + df["longitude"].round(4).astype(str)

region_locations = df[df["region"] == region]["location_label"].unique().tolist()

with col1:
    location = st.selectbox("Select Location (Lat, Long):", ["All Locations"] + region_locations)

with col2:
    crop = st.selectbox("Select Crop Type:", df["crop_type"].unique())

with col3:
    fertilizer = st.selectbox("Select Fertilizer:", df["fertilizer_type"].unique())

with col3:
    feature_x = st.selectbox(
        "Select X-Axis Feature (for visualization only):",
        ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]
    )

filtered_df = df[
    (df["region"] == region) &
    (df["crop_type"] == crop) &
    (df["fertilizer_type"] == fertilizer)
].copy()

if location != "All Locations":
    filtered_df = filtered_df[filtered_df["location_label"] == location]


# ===============================
# 3. Soil Moisture Column Detection
# ===============================
soil_col = "soil_moisture_%" if "soil_moisture_%" in df.columns else "soil_moisture"
st.caption(f"Using soil moisture column: **{soil_col}**")


# ===============================
# 3b. Fine-Grained Soil Moisture Classification (FIXED)
# ===============================
bins = [0, 10, 20, 30, 40, 50, 60, 100]
labels = [
    "Very Dry (0–10%)",
    "Dry (10–20%)",
    "Moderate Dry (20–30%)",
    "Optimal Low (30–40%)",
    "Optimal High (40–50%)",
    "Wet (50–60%)",
    "Very Wet (>60%)"
]

filtered_df["Soil_Moisture_Level"] = pd.cut(
    filtered_df[soil_col],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ===============================
# 📊 Visualization
# ===============================
st.subheader("📊 Soil Moisture Relationship Visualization")

if len(filtered_df) > 0:
    fig_vis = px.scatter(
        filtered_df,
        x=feature_x,
        y=soil_col,
        color="Soil_Moisture_Level",
        title=f"Soil Moisture vs {feature_x}",
        labels={soil_col: "Soil Moisture (%)"}
    )
    st.plotly_chart(fig_vis, use_container_width=True)
    st.dataframe(filtered_df[[feature_x, soil_col, "Soil_Moisture_Level"]])
else:
    st.warning("⚠ No data available.")


# ===============================
# 4. Model Preparation
# ===============================
X = filtered_df[["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]].copy()
X["crop_type_encoded"] = LabelEncoder().fit_transform(filtered_df["crop_type"])
X["fertilizer_type_encoded"] = LabelEncoder().fit_transform(filtered_df["fertilizer_type"])
y = filtered_df[soil_col]

if len(filtered_df) < 5:
    X_train = X_test = X
    y_train = y_test = y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 5. Train Models
# ===============================
dt_model = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

nn_model = MLPRegressor(hidden_layer_sizes=(nn_layer_size, nn_layer_size), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)


# ===============================
# 8. Latest Prediction (UPDATED LOGIC)
# ===============================
latest_features = X.tail(1)

dt_latest = dt_model.predict(latest_features)[0]
nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

avg = (dt_latest + nn_latest) / 2

if 10 <= avg < 20:
    condition = "🚰 Water Needed (10–20%)"
    bar_color = "red"
elif avg < 10:
    condition = "🌵 Very Dry (<10%)"
    bar_color = "orange"
elif avg < 30:
    condition = "🌾 Slightly Dry (20–30%)"
    bar_color = "yellow"
elif avg < 50:
    condition = "🌱 Optimal"
    bar_color = "green"
elif avg < 60:
    condition = "💧 Wet"
    bar_color = "blue"
else:
    condition = "🌊 Very Wet (>60%)"
    bar_color = "darkblue"

st.progress(int(avg))
st.markdown(f"<p style='color:{bar_color}; font-size:18px;'>{condition}</p>", unsafe_allow_html=True)
