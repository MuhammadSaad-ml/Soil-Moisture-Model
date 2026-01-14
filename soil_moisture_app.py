import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from streamlit_plotly_events import plotly_events


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
with col1:
    crop = st.selectbox("Select Crop Type:", df["crop_type"].unique())
with col2:
    fertilizer = st.selectbox("Select Fertilizer:", df["fertilizer_type"].unique())
with col3:
    feature_x = st.selectbox(
        "Select X-Axis Feature:",
        ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]
    )

filtered_df = df[
    (df["region"] == region) &
    (df["crop_type"] == crop) &
    (df["fertilizer_type"] == fertilizer)
].copy()


# ===============================
# 3. Soil Moisture Detection
# ===============================
if "soil_moisture_%" in df.columns:
    soil_col = "soil_moisture_%"
elif "soil_moisture" in df.columns:
    soil_col = "soil_moisture"
else:
    soil_col = [c for c in df.columns if "moisture" in c.lower()][0]

st.caption(f"Using soil moisture column: **{soil_col}**")

bins = [0, 30, 60, 100]
labels = ["Dry", "Optimal", "Wet"]

filtered_df["Soil_Moisture_Level"] = pd.cut(
    filtered_df[soil_col], bins=bins, labels=labels
)


# ===============================
# 📊 Visualization (CLICK ENABLED)
# ===============================
st.subheader("📊 Soil Moisture Relationship Visualization")

st.session_state.setdefault("selected_row", None)

if len(filtered_df) > 0:
    fig_vis = px.scatter(
        filtered_df,
        x=feature_x,
        y=soil_col,
        color="Soil_Moisture_Level",
        color_discrete_map={"Dry": "red", "Optimal": "green", "Wet": "blue"},
        title=f"Soil Moisture vs {feature_x}"
    )

    fig_vis.update_traces(marker=dict(size=11, opacity=0.75))

    selected_points = plotly_events(
        fig_vis,
        click_event=True,
        hover_event=False,
        select_event=False,
        key="soil_chart"
    )

    if selected_points:
        idx = selected_points[0]["pointIndex"]
        st.session_state["selected_row"] = filtered_df.iloc[idx]
    else:
        st.session_state["selected_row"] = None

    st.dataframe(
        filtered_df[[feature_x, soil_col, "Soil_Moisture_Level"]],
        use_container_width=True
    )
else:
    st.warning("⚠ No data available.")


# ===============================
# 🗺️ INTERACTIVE MAP (ACTION FILTER)
# ===============================
st.subheader("🗺️ Field Location Map")

if {"latitude", "longitude"}.issubset(filtered_df.columns):

    # If chart point clicked → show only that location
    if st.session_state["selected_row"] is not None:
        map_df = pd.DataFrame([{
            "latitude": st.session_state["selected_row"]["latitude"],
            "longitude": st.session_state["selected_row"]["longitude"]
        }])

        st.map(map_df)
        st.caption("📍 Showing location selected from chart.")

    # Else → show entire region
    else:
        map_df = filtered_df[["latitude", "longitude"]].dropna()
        st.map(map_df)
        st.caption("📍 Showing all locations in selected region.")

else:
    st.error("❌ Latitude/Longitude columns missing.")


# ===============================
# 4. Model Preparation
# ===============================
model_features = ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]
X = filtered_df[model_features].copy()

le_crop = LabelEncoder()
le_fert = LabelEncoder()

X["crop_type_encoded"] = le_crop.fit_transform(filtered_df["crop_type"])
X["fertilizer_type_encoded"] = le_fert.fit_transform(filtered_df["fertilizer_type"])

y = filtered_df[soil_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 5. Train Models
# ===============================
dt_model = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

nn_model = MLPRegressor(
    hidden_layer_sizes=(nn_layer_size, nn_layer_size),
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)


# ===============================
# 6. Model Performance
# ===============================
st.subheader("📉 Model Accuracy Comparison")

col1, col2 = st.columns(2)

with col1:
    dt_df = pd.DataFrame({"Actual": y_test, "Predicted": dt_pred})
    dt_df["Error"] = abs(dt_df["Actual"] - dt_df["Predicted"])
    st.dataframe(dt_df, use_container_width=True)

with col2:
    nn_df = pd.DataFrame({"Actual": y_test, "Predicted": nn_pred})
    nn_df["Error"] = abs(nn_df["Actual"] - nn_df["Predicted"])
    st.dataframe(nn_df, use_container_width=True)


# ===============================
# 7. Latest Prediction
# ===============================
latest_features = X.tail(1)

dt_latest = dt_model.predict(latest_features)[0]
nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

st.subheader("💧 Latest Soil Moisture Prediction")

st.metric("🌳 Decision Tree", f"{dt_latest:.2f}%")
st.metric("🤖 Neural Network", f"{nn_latest:.2f}%")
