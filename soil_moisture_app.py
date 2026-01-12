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
# 3. Soil Moisture Detection
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
# 5. Traditional ML Models
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

dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))

dt_mae = mean_absolute_error(y_test, dt_pred)
nn_mae = mean_absolute_error(y_test, nn_pred)


# ===============================
# 🌱 6. TinyML Model (NEW – EDGE SIMULATION)
# ===============================
st.markdown("## 🌱 TinyML Model (Edge Device Simulation)")

tiny_features = ["temperature_C", "humidity_%"]
X_tiny = filtered_df[tiny_features]
y_tiny = filtered_df[soil_col]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_tiny, y_tiny, test_size=test_size, random_state=42
)

tiny_scaler = StandardScaler()
X_train_t = tiny_scaler.fit_transform(X_train_t)
X_test_t = tiny_scaler.transform(X_test_t)

tiny_model = DecisionTreeRegressor(
    max_depth=2,
    min_samples_leaf=10,
    random_state=42
)

tiny_model.fit(X_train_t, y_train_t)
tiny_pred = tiny_model.predict(X_test_t)

tiny_rmse = np.sqrt(mean_squared_error(y_test_t, tiny_pred))
tiny_mae = mean_absolute_error(y_test_t, tiny_pred)


# ===============================
# 7. Model Comparison
# ===============================
st.markdown("## 📊 Traditional ML vs TinyML Comparison")

compare_df = pd.DataFrame({
    "Model": ["Decision Tree", "Neural Network", "TinyML"],
    "MAE": [dt_mae, nn_mae, tiny_mae],
    "RMSE": [dt_rmse, nn_rmse, tiny_rmse]
})

st.dataframe(compare_df)

fig_compare = px.bar(compare_df, x="Model", y="RMSE", color="Model")
st.plotly_chart(fig_compare, use_container_width=True)


# ===============================
# 8. Explanation Section
# ===============================
st.markdown("""
## 🧠 Traditional Machine Learning vs TinyML

| Aspect | Traditional ML | TinyML |
|------|---------------|-------|
| Deployment | Cloud / PC | Microcontroller |
| Model Size | Large | Very Small |
| Accuracy | High | Moderate |
| Power | High | Ultra-low |
| Latency | Medium | Real-time |
| Example | Dashboards | Field sensors |

✔️ Your app now demonstrates **both paradigms**
""")


# ===============================
# 9. Latest Predictions
# ===============================
latest_features = X.tail(1)

dt_latest = dt_model.predict(latest_features)[0]
nn_latest = nn_model.predict(scaler.transform(latest_features))[0]

latest_tiny = tiny_scaler.transform(
    latest_features[tiny_features]
)
tiny_latest = tiny_model.predict(latest_tiny)[0]

st.markdown("## 💧 Latest Soil Moisture Prediction")

st.markdown(f"🌳 **Decision Tree:** {dt_latest:.2f}%")
st.markdown(f"🤖 **Neural Network:** {nn_latest:.2f}%")
st.markdown(f"🌱 **TinyML:** {tiny_latest:.2f}%")
