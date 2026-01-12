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

test_size = st.sidebar.slider(
    "Train/Test Split (Test Size)",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)

tree_depth = st.sidebar.slider(
    "Decision Tree Max Depth",
    min_value=2,
    max_value=15,
    value=5
)

nn_layer_size = st.sidebar.slider(
    "Neural Network Hidden Layer Size",
    min_value=16,
    max_value=128,
    value=50,
    step=16
)


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
# 3. Soil Moisture Column Detection
# ===============================
if "soil_moisture_%" in df.columns:
    soil_col = "soil_moisture_%"
elif "soil_moisture" in df.columns:
    soil_col = "soil_moisture"
else:
    candidates = [c for c in df.columns if "moisture" in c.lower()]
    if not candidates:
        st.error("❌ No soil moisture column found.")
        st.stop()
    soil_col = candidates[0]

bins = [0, 30, 60, 100]
labels = ["Dry", "Optimal", "Wet"]

filtered_df["Soil_Moisture_Level"] = pd.cut(
    filtered_df[soil_col], bins=bins, labels=labels
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
        color_discrete_map={"Dry": "red", "Optimal": "green", "Wet": "blue"}
    )
    st.plotly_chart(fig_vis, use_container_width=True)
    st.dataframe(filtered_df[[feature_x, soil_col, "Soil_Moisture_Level"]])
else:
    st.warning("No data available.")


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
# 6. Actual vs Predicted
# ===============================
st.subheader("📉 Model Accuracy Comparison")

dt_df = pd.DataFrame({"Actual": y_test.values, "Predicted": dt_pred})
dt_df["Error"] = abs(dt_df["Actual"] - dt_df["Predicted"])

nn_df = pd.DataFrame({"Actual": y_test.values, "Predicted": nn_pred})
nn_df["Error"] = abs(nn_df["Actual"] - nn_df["Predicted"])


# ===============================
# 🔴 NEW SECTION: Highest Error Analysis (ONLY ADDITION)
# ===============================
st.markdown("## 🚨 Highest Prediction Errors")

combined_errors = pd.concat([
    dt_df.assign(Model="Decision Tree"),
    nn_df.assign(Model="Neural Network")
])

top_errors = combined_errors.sort_values("Error", ascending=False).head(10)

st.markdown("### ❌ Top 10 Worst Predictions (Actual vs Predicted)")
st.dataframe(top_errors, use_container_width=True)

fig_err = px.bar(
    top_errors,
    x=top_errors.index,
    y="Error",
    color="Model",
    title="Highest Absolute Prediction Errors"
)
st.plotly_chart(fig_err, use_container_width=True)


# ===============================
# 7. Model Performance Summary
# ===============================
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))

st.markdown("## 📐 Model Performance Metrics")
st.metric("🌳 DT RMSE", f"{dt_rmse:.2f}")
st.metric("🤖 NN RMSE", f"{nn_rmse:.2f}")


# ===============================
# 8. Latest Predictions
# ===============================
latest_features = X.tail(1)

dt_latest = dt_model.predict(latest_features)[0]
nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

st.markdown("## 💧 Latest Soil Moisture Prediction")
st.markdown(f"🌳 Decision Tree: **{dt_latest:.2f}%**")
st.markdown(f"🤖 Neural Network: **{nn_latest:.2f}%**")
