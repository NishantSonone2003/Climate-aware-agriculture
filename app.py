import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Climate-Aware Agriculture Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to bottom, #d4f4dd, #f5f5dc);
}

/* Main Header */
.main-header {
    background: linear-gradient(90deg, #2E8B57, #3CB371);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    background-color: #2E8B57;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    width: 100%;
    height: 50px;
}

/* Result Cards */
.result-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2E8B57;
    margin-bottom: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size: 14px;
}
/* All normal text */
p, div, label {
    color: #1B4332 !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #0B3D0B !important;
    font-weight: bold;
}

/* Subheaders like Data Distribution Analysis */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #0B3D0B !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #2E8B57;
}

[data-testid="metric-container"] label {
    color: #1B4332 !important;
    font-weight: bold;
}

[data-testid="metric-container"] div {
    color: #000000 !important;
    font-size: 24px;
    font-weight: bold;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #1B4332 !important;
}

/* Keep every input reachable on shorter screens */
section[data-testid="stSidebar"] {
    min-width: 340px !important;
    background: linear-gradient(180deg, #eef8f0 0%, #f8fbf2 100%) !important;
    border-right: 1px solid #cfe3d3;
}

section[data-testid="stSidebar"] > div {
    overflow-y: auto;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #c9dfce;
    border-radius: 10px;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #ffffff !important;
    color: #123d24 !important;
}

section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background-color: #ffffff !important;
    border-color: #9fc7aa !important;
}

section[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #102f1c !important;
    -webkit-text-fill-color: #102f1c !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] span {
    color: #173f27 !important;
}

section[data-testid="stSidebar"] button[kind="primary"] p,
section[data-testid="stSidebar"] button[kind="primary"] span {
    color: #ffffff !important;
}

.sidebar-guide {
    background: #ffffff;
    border-left: 5px solid #2E8B57;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 2px 0 10px;
    line-height: 1.65;
    box-shadow: 0 4px 14px rgba(27, 67, 50, 0.08);
}

.sidebar-guide strong {
    display: inline-block;
    color: #0B3D0B !important;
    font-size: 1rem;
    margin-bottom: 3px;
}

.sidebar-guide small {
    display: inline-block;
    margin-top: 6px;
    color: #52705e !important;
    line-height: 1.35;
}

/* Histogram section title */
.stSubheader {
    color: #0B3D0B !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="main-header">
    <h1>🌾 Climate-Aware Agriculture Prediction App</h1>
    <h4>AI-Powered Crop Recommendation & Yield Prediction System</h4>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
color:#1B4332;
font-size:20px;
font-weight:600;
padding:15px;
background-color:rgba(255,255,255,0.7);
border-radius:10px;
">
This application predicts:<br><br>

✅ Most Suitable Crop<br><br>

✅ Expected Crop Yield (tons/hectare)<br><br>

using climate and soil conditions.
</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("sample_climate_agri.csv")

df = load_data()

# ==================================================
# PREPARE CROP MODEL
# ==================================================

X_crop = df.drop(["crop", "state", "yield_t_ha"], axis=1)
y_crop = df["crop"]

crop_encoder = LabelEncoder()
y_crop_encoded = crop_encoder.fit_transform(y_crop)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_crop,
    y_crop_encoded,
    test_size=0.2,
    random_state=42
)

crop_model = LogisticRegression(max_iter=500)
crop_model.fit(Xc_train, yc_train)

# ==================================================
# PREPARE YIELD MODEL
# ==================================================

X_yield = df.drop(["yield_t_ha", "crop", "state"], axis=1)
y_yield = df["yield_t_ha"]

Xy_train, Xy_test, yy_train, yy_test = train_test_split(
    X_yield,
    y_yield,
    test_size=0.2,
    random_state=42
)

yield_model = LinearRegression()
yield_model.fit(Xy_train, yy_train)

# ==================================================
# SIDEBAR INPUTS
# ==================================================

st.sidebar.header("Prediction Inputs")

inputs = {}

st.sidebar.markdown(
    """
    <div class="sidebar-guide">
        <strong>How to use this panel</strong><br>
        <span>1. Open each numbered group.</span><br>
        <span>2. Enter your farm's current values.</span><br>
        <span>3. Click <b>Predict</b> below.</span><br>
        <small>The pre-filled values are dataset averages and can be changed.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

input_groups = {
    "1 · Climate — air & rainfall": ["year", "avg_temp_c", "rainfall_mm", "co2_ppm"],
    "2 · Soil — moisture & fertilizer": ["soil_moisture", "fertilizer_kg_ha"],
    "3 · Farm — technology & pests": ["tech_index", "pest_index"],
}

input_labels = {
    "year": "Year",
    "avg_temp_c": "Average temperature (°C)",
    "rainfall_mm": "Rainfall (mm)",
    "co2_ppm": "CO₂ concentration (ppm)",
    "soil_moisture": "Soil moisture",
    "fertilizer_kg_ha": "Fertilizer (kg/ha)",
    "tech_index": "Technology index",
    "pest_index": "Pest pressure index",
}

for group_name, columns in input_groups.items():
    with st.sidebar.expander(group_name, expanded=group_name.startswith("1")):
        for col in columns:
            if col == "year":
                inputs[col] = st.number_input(
                    input_labels[col],
                    min_value=2000,
                    max_value=2100,
                    value=2025,
                )
            else:
                inputs[col] = st.number_input(
                    input_labels[col],
                    value=float(df[col].mean()),
                )

input_df = pd.DataFrame([inputs])

# ==================================================
# PREDICTION
# ==================================================

if st.sidebar.button("Predict", type="primary", use_container_width=True):

    # Crop Prediction
    crop_pred = crop_model.predict(input_df)[0]
    crop_name = crop_encoder.inverse_transform([crop_pred])[0]

    # Yield Prediction
    yield_pred = yield_model.predict(input_df)[0]

    # ==================================================
    # RESULTS
    # ==================================================

    st.markdown(f"""
    <div class="result-card">
        <h2>🌱 Recommended Crop</h2>
        <h1>{crop_name}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <h2>📊 Expected Yield</h2>
        <h1>{yield_pred:.2f} tons/hectare</h1>
    </div>
    """, unsafe_allow_html=True)

    # ==================================================
    # METRICS
    # ==================================================

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
<div style='background:white;
padding:15px;
border-radius:10px;
border-left:5px solid #2E8B57;
margin-bottom:10px;'>

<h3 style='color:#0B3D0B;'>🌾 Predicted Crop</h3>
<h2 style='color:black;'>{crop_name}</h2>

</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div style='background:white;
padding:15px;
border-radius:10px;
border-left:5px solid #2E8B57;
margin-bottom:10px;'>

<h3 style='color:#0B3D0B;'>📈 Predicted Yield</h3>
<h2 style='color:black;'>{yield_pred:.2f} t/ha</h2>

</div>
""", unsafe_allow_html=True)

    # ==================================================
    # HISTOGRAMS
    # ==================================================

    st.markdown("""
<h2 style='color:#0B3D0B; background:white;
padding:10px; border-radius:10px;'>
📊 Data Distribution Analysis
</h2>
""", unsafe_allow_html=True)

    numeric_cols = []

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

        except:
            pass

    for col in numeric_cols:

        if col in ["state", "crop"]:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            df[col].dropna(),
            bins=20,
            color="lightgreen",
            edgecolor="black"
        )

        if col in inputs:
            ax.axvline(
                inputs[col],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Your Input"
            )

        if col == "yield_t_ha":
            ax.axvline(
                yield_pred,
                color="blue",
                linestyle="--",
                linewidth=2,
                label="Predicted Yield"
            )

        ax.set_title(
            f"Distribution of {col}",
            fontsize=12
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<hr>
<div class="footer">
Developed by Nishant Sonone | BE Computer Science Engineering | 2026
</div>
""", unsafe_allow_html=True)
