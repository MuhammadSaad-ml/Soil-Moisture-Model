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
# ⚙️ Model Tuning Controls (NEW)
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
# 2. Filters (Region, Crop, Fertilizer)
# ===============================
col0, col1, col2, col3 = st.columns(4)

with col0:
    region = st.selectbox("Select Region:", df["region"].unique())

# Create location label from lat/long
df["location_label"] = (
    df["latitude"].round(4).astype(str) + ", " +
    df["longitude"].round(4).astype(str)
)

# Locations available inside selected region
region_locations = (
    df[df["region"] == region]["location_label"]
    .unique()
    .tolist()
)

with col1:
    location = st.selectbox(
        "Select Location (Lat, Long):",
        options=["All Locations"] + region_locations
    )


with col2:
    crop = st.selectbox("Select Crop Type:", df["crop_type"].unique())

with col3:
    fertilizer = st.selectbox("Select Fertilizer:", df["fertilizer_type"].unique())


with col3:
    feature_x = st.selectbox(
        "Select X-Axis Feature (for visualization only):",
        ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]
    )


# Apply filters
filtered_df = df[
    (df["region"] == region) &
    (df["crop_type"] == crop) &
    (df["fertilizer_type"] == fertilizer)
].copy()

# Apply location filter if selected
if location != "All Locations":
    filtered_df = filtered_df[
        filtered_df["location_label"] == location
    ]



# ===============================
# 3. Soil Moisture Column Detection + Classification
# ===============================
if "soil_moisture_%" in df.columns:
    soil_col = "soil_moisture_%"
elif "soil_moisture" in df.columns:
    soil_col = "soil_moisture"
else:
    candidates = [c for c in df.columns if "moisture" in c.lower()]
    if len(candidates) == 0:
        st.error("❌ No soil moisture column found in the dataset.")
        st.stop()
    soil_col = candidates[0]

st.caption(f"Using soil moisture column: **{soil_col}**")

# ===============================
# 3b. Fine-Grained Soil Moisture Classification (NEW)
# ===============================

bins = [0, 10, 20, 30, 40, 50, 60, 100]
labels={
    "Very Dry (0–10%)": "#8B0000",
    "Dry (10–20%)": "#FF4500",
    "Moderate Dry (20–30%)": "#FFA500",
    "Optimal Low (30–40%)": "#9ACD32",
    "Optimal High (40–50%)": "#228B22",
    "Wet (50–60%)": "#1E90FF",
    "Very Wet (>60%)": "#00008B"
}


filtered_df["Soil_Moisture_Level"] = pd.cut(
    filtered_df[soil_col],
    bins=bins,
    labels=labels,
    include_lowest=True
)



# ===============================
# 📊 Visualization Section
# ===============================
st.subheader("📊 Soil Moisture Relationship Visualization")

if len(filtered_df) > 0:
    fig_vis = px.scatter(
        filtered_df,
        x=feature_x,
        y=soil_col,
        color="Soil_Moisture_Level",
        title=f"Soil Moisture vs {feature_x}",
        labels={
            feature_x: feature_x.replace("_", " ").title(),
            soil_col: "Soil Moisture (%)"
        },
        color_discrete_map={"Dry": "red", "Optimal": "green", "Wet": "blue"}
    )

    fig_vis.update_traces(marker=dict(size=11, opacity=0.75))
    fig_vis.update_layout(height=450)

    st.plotly_chart(fig_vis, use_container_width=True)

    st.markdown("### 🔍 Data Preview")
    st.dataframe(
        filtered_df[[feature_x, soil_col, "Soil_Moisture_Level"]],
        use_container_width=True
    )
else:
    st.warning("⚠ No data available for the selected filters.")


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
# 🚨 Not enough data safeguard
if len(filtered_df) < 5:
    st.warning(
        "⚠ Not enough data for model training at this location. "
        "Showing predictions using available data only."
    )

    # Use entire data for prediction only
    X_train = X
    y_train = y
    X_test = X
    y_test = y
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=test_size, random_state=42
# )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 5. Train Models
# ===============================
dt_model = DecisionTreeRegressor(
    max_depth=tree_depth,
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

nn_model = MLPRegressor(
    hidden_layer_sizes=(nn_layer_size, nn_layer_size),
    max_iter=1000,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)
nn_pred = nn_model.predict(X_test_scaled)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))


# ===============================
# 6. Actual vs Predicted Comparison
# ===============================
st.subheader("📉 Model Accuracy Comparison: Actual vs Predicted Soil Moisture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌳 Decision Tree")
    dt_df = pd.DataFrame({"Actual": y_test.values, "Predicted": dt_pred})
    dt_df["Error"] = abs(dt_df["Actual"] - dt_df["Predicted"])
    st.dataframe(dt_df, use_container_width=True)

    fig_dt = px.scatter(
        dt_df,
        x="Actual",
        y="Predicted",
        color="Error",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_dt, use_container_width=True)

with col2:
    st.markdown("### 🤖 Neural Network")
    nn_df = pd.DataFrame({"Actual": y_test.values, "Predicted": nn_pred})
    nn_df["Error"] = abs(nn_df["Actual"] - nn_df["Predicted"])
    st.dataframe(nn_df, use_container_width=True)

    fig_nn = px.scatter(
        nn_df,
        x="Actual",
        y="Predicted",
        color="Error",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_nn, use_container_width=True)


# ===============================
# 🔴 NEW SECTION (ONLY ADDITION)
# ===============================
st.markdown("## 🚨 Highest Prediction Errors (Actual vs Predicted)")

combined_errors = pd.concat([
    dt_df.assign(Model="Decision Tree"),
    nn_df.assign(Model="Neural Network")
])

highest_errors = combined_errors.sort_values("Error", ascending=False).head(10)

st.dataframe(highest_errors, use_container_width=True)

fig_high_err = px.bar(
    highest_errors,
    x=highest_errors.index,
    y="Error",
    color="Model",
    title="Top 10 Highest Prediction Errors"
)
st.plotly_chart(fig_high_err, use_container_width=True)


# ===============================
# 7. Model Performance Summary
# ===============================
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_mse = mean_squared_error(y_test, nn_pred)

dt_std = np.std(abs(y_test - dt_pred))
nn_std = np.std(abs(y_test - nn_pred))

st.markdown("## 📐 Model Performance Metrics")

colA, colB, colC, colD = st.columns(4)
colA.metric("🌳 DT – MAE", f"{dt_mae:.2f}")
colB.metric("🌳 DT – MSE", f"{dt_mse:.2f}")
colC.metric("🌳 DT – RMSE", f"{dt_rmse:.2f}")
colD.metric("🌳 DT – Std Dev", f"{dt_std:.2f}")

colA2, colB2, colC2, colD2 = st.columns(4)
colA2.metric("🤖 NN – MAE", f"{nn_mae:.2f}")
colB2.metric("🤖 NN – MSE", f"{nn_mse:.2f}")
colC2.metric("🤖 NN – RMSE", f"{nn_rmse:.2f}")
colD2.metric("🤖 NN – Std Dev", f"{nn_std:.2f}")


# ===============================
# ℹ️ Metric Explanation
# ===============================
st.markdown("""
### ℹ️ What Do These Numbers Mean?

| Metric | Meaning | Why it Matters |
|------|--------|---------------|
| **MAE** | Average prediction error | Easy to interpret |
| **MSE** | Penalizes large mistakes | Detects instability |
| **RMSE** | Error in real units (%) | Most intuitive |
| **Std Dev** | Error consistency | Lower = more reliable |
""")


# ===============================
# 8. Latest Predictions
# ===============================
latest_features = X.tail(1)

dt_latest = dt_model.predict(latest_features)[0]
nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

st.markdown("---")
st.subheader("💧 Latest Soil Moisture Prediction")

colA, colB = st.columns(2)
colA.markdown(
    f"<h3 style='color:#2DBBCC;'>🌳 {dt_latest:.2f}%</h3>",
    unsafe_allow_html=True
)
colB.markdown(
    f"<h3 style='color:#2DBBCC;'>🤖 {nn_latest:.2f}%</h3>",
    unsafe_allow_html=True
)

avg = (dt_latest + nn_latest) / 2

if avg < 30:
    condition = "🌵 Dry — Needs Water"
    bar_color = "red"
elif avg < 60:
    condition = "🌾 Optimal"
    bar_color = "green"
else:
    condition = "💧 Too Wet"
    bar_color = "blue"

st.progress(int(avg))
st.markdown(
    f"<p style='color:{bar_color}; font-size:18px;'>{condition}</p>",
    unsafe_allow_html=True
)






