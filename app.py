import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="UK Road Accident Severity Predictor", page_icon="🚗", layout="centered")

class AccidentNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_artifacts():
    scaler          = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    ordinal_maps    = joblib.load("ordinal_mappings.pkl")
    threshold       = joblib.load("threshold.pkl")
    input_dim       = joblib.load("input_dim.pkl")
    model = AccidentNet(input_dim)
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()
    return model, scaler, feature_columns, ordinal_maps, threshold

model, scaler, feature_columns, ordinal_maps, threshold = load_artifacts()

st.title("🚗 UK Road Accident Severity Predictor")
st.markdown("Enter accident conditions to predict **Slight** or **Serious/Fatal** outcome.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("📍 Location & Time")
    year        = st.selectbox("Year", list(range(2005, 2026)), index=12)
    hour        = st.slider("Hour of Day", 0, 23, 8)
    month       = st.selectbox("Month", list(range(1, 13)), index=0)
    in_scotland = st.selectbox("In Scotland?", ["No", "Yes"])
    urban_rural = st.selectbox("Area Type", ["Urban", "Rural"])
    is_weekend  = st.selectbox("Day Type", ["Weekday", "Weekend"])

with col2:
    st.subheader("🛣️ Road Conditions")
    speed_limit  = st.selectbox("Speed Limit (mph)", [20, 30, 40, 50, 60, 70], index=1)
    road_type    = st.selectbox("Road Type", ["Single carriageway","Dual carriageway","Roundabout","One way street","Slip road","Unknown"])
    road_class_1 = st.selectbox("1st Road Class", ["A","B","C","Motorway","A(M)","Unclassified"])
    road_class_2 = st.selectbox("2nd Road Class", ["Not at Junction","A","B","C","Motorway","A(M)","Unclassified"])
    light        = st.selectbox("Light Conditions", ["Daylight","Darkness - lights lit","Darkness - lights unlit","Darkness - no lighting","Darkness - lighting unknown","Unknown"])
    weather      = st.selectbox("Weather", ["Fine no high winds","Raining no high winds","Snowing no high winds","Fine + high winds","Raining + high winds","Snowing + high winds","Fog or mist","Other","Unknown"])
    road_surface = st.selectbox("Road Surface", ["Dry","Wet or damp","Snow","Frost or ice","Flood over 3cm deep","Unknown"])

st.subheader("🔀 Junction & Pedestrian")
col3, col4 = st.columns(2)
with col3:
    junction_detail  = st.selectbox("Junction Detail", ["Not at junction or within 20 metres","Roundabout","Mini-roundabout","T or staggered junction","Slip road","Crossroads","More than 4 arms (not roundabout)","Private drive or entrance","Other junction"])
    junction_control = st.selectbox("Junction Control", ["Not at junction or within 20 metres","Authorised person","Auto traffic signal","Give way or uncontrolled","Stop sign","Unknown"])
with col4:
    ped_human    = st.selectbox("Pedestrian Crossing - Human Control", ["None","Present"])
    ped_physical = st.selectbox("Pedestrian Crossing - Physical Facilities", ["None","Present"])
    season       = st.selectbox("Season", ["Spring","Summer","Autumn","Winter"])

st.divider()

def build_input():
    is_rush  = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
    is_night = 1 if hour >= 20 or hour <= 6 else 0
    row = {
        "Year": year, "InScotland": 1 if in_scotland=="Yes" else 0,
        "Urban_or_Rural_Area": ordinal_maps["Urban_or_Rural_Area"][urban_rural],
        "1st_Road_Class":      ordinal_maps["1st_Road_Class"][road_class_1],
        "2nd_Road_Class":      ordinal_maps["2nd_Road_Class"][road_class_2],
        "Road_Type":           ordinal_maps["Road_Type"][road_type],
        "Speed_limit": speed_limit,
        "Light_Conditions":        ordinal_maps["Light_Conditions"].get(light, 0),
        "Road_Surface_Conditions": ordinal_maps["Road_Surface_Conditions"].get(road_surface, 0),
        "Pedestrian_Crossing-Human_Control":       1 if ped_human=="Present" else 0,
        "Pedestrian_Crossing-Physical_Facilities": 1 if ped_physical=="Present" else 0,
        "Hour": hour, "Month": month,
        "is_rush_hour": is_rush, "is_night": is_night,
        "is_weekend": 1 if is_weekend=="Weekend" else 0,
    }
    df_row = pd.DataFrame([{col: 0 for col in feature_columns}])
    for k, v in row.items():
        if k in df_row.columns:
            df_row[k] = v
    for col_name, val in [
        ("Weather_Conditions_" + weather, 1),
        ("Junction_Detail_" + junction_detail, 1),
        ("Junction_Control_" + junction_control, 1),
        ("Season_" + season, 1)
    ]:
        if col_name in df_row.columns:
            df_row[col_name] = val
    return df_row[feature_columns]

if st.button("Predict Severity", use_container_width=True, type="primary"):
    input_df = build_input()
    scaled   = scaler.transform(input_df).astype(np.float32)
    tensor   = torch.tensor(scaled)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    pred = 1 if prob >= threshold else 0
    st.divider()
    if pred == 1:
        conf = str(round(prob * 100, 1))
        thr  = str(round(threshold, 3))
        st.error("### Serious or Fatal Accident Predicted\n\n**Confidence:** " + conf + "%\n\n**Threshold:** " + thr)
    else:
        conf = str(round((1 - prob) * 100, 1))
        thr  = str(round(threshold, 3))
        st.success("### Slight Accident Predicted\n\n**Confidence:** " + conf + "%\n\n**Threshold:** " + thr)
    with st.expander("Prediction Details"):
        st.metric("Raw probability (Serious/Fatal)", str(round(prob, 4)))
        st.metric("Decision threshold", str(round(threshold, 4)))
        st.metric("Prediction", "Serious/Fatal" if pred == 1 else "Slight")

st.divider()
st.caption("UK Road Safety Project · STATS19 (2005-2017) · PyTorch Neural Network · 2M rows")