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
This model predicts **soil moisture** based on temperature, humidity, rainfall, soil pH, crop, and fertilizer type.
""")


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

with col1:
    crop = st.selectbox("Select Crop Type:", df["crop_type"].unique())

with col2:
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


# ===============================
# 3. Soil Moisture Column Detection + Classification
# ===============================
# (1) Detect soil moisture column safely
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

# (2) Assign moisture level categories
bins = [0, 30, 60, 100]
labels = ["Dry", "Optimal", "Wet"]

filtered_df["Soil_Moisture_Level"] = pd.cut(
    filtered_df[soil_col], bins=bins, labels=labels
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
    st.dataframe(filtered_df[[feature_x, soil_col, "Soil_Moisture_Level"]], use_container_width=True)

else:
    st.warning("⚠ No data available for the selected filters.")


# ===============================
# 4. Model Preparation
# ===============================
model_features = ["temperature_C", "humidity_%", "rainfall_mm", "soil_pH"]
X = filtered_df[model_features].copy()

# Encode categories
le_crop = LabelEncoder()
le_fert = LabelEncoder()

X["crop_type_encoded"] = le_crop.fit_transform(filtered_df["crop_type"])
X["fertilizer_type_encoded"] = le_fert.fit_transform(filtered_df["fertilizer_type"])

y = filtered_df[soil_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling for NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 5. Train Models
# ===============================
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

nn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
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

    fig_dt = px.scatter(dt_df, x="Actual", y="Predicted", color="Error", color_continuous_scale="Viridis")
    st.plotly_chart(fig_dt, use_container_width=True)

with col2:
    st.markdown("### 🤖 Neural Network")
    nn_df = pd.DataFrame({"Actual": y_test.values, "Predicted": nn_pred})
    nn_df["Error"] = abs(nn_df["Actual"] - nn_df["Predicted"])
    st.dataframe(nn_df, use_container_width=True)

    fig_nn = px.scatter(nn_df, x="Actual", y="Predicted", color="Error", color_continuous_scale="Viridis")
    st.plotly_chart(fig_nn, use_container_width=True)


# ===============================
# 7. Model Performance Summary
# ===============================
st.markdown("---")
st.subheader("📊 Model Performance Overview")

col3, col4 = st.columns(2)
col3.metric("🌳 Decision Tree RMSE", f"{dt_rmse:.2f}%")
col4.metric("🤖 Neural Network RMSE", f"{nn_rmse:.2f}%")

if nn_rmse < dt_rmse:
    st.success("✅ Neural Network is more accurate.")
else:
    st.warning("⚠ Decision Tree performed better.")


# ===============================
# 8. Latest Predictions
# ===============================
latest_features = X.tail(1)

dt_latest = dt_model.predict(latest_features)[0]
nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

st.markdown("---")
st.subheader("💧 Latest Soil Moisture Prediction")

col5, col6 = st.columns(2) with col5: st.markdown( f"**🌳 Decision Tree Prediction:** <h2 style='color:#2DBBCC;'>{pred_dt_latest:.2f}%</h2>", unsafe_allow_html=True ) with col6: st.markdown( f"**🤖 Neural Network Prediction:** <h2 style='color:#2DBBCC;'>{pred_nn_latest:.2f}%</h2>", unsafe_allow_html=True )
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
st.markdown(f"<p style='color:{bar_color}; font-size:18px;'>{condition}</p>", unsafe_allow_html=True)


st.info("""
### ℹ️ RMSE Meaning  
Lower RMSE = better prediction accuracy.
""")

