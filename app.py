"""Streamlit interface for the network intrusion detection demo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from src.model_utils import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, DATA_PATH, load_or_train

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Network Intrusion Detection System")
st.caption("Machine-learning demonstration using network-flow features")

st.warning(
    "Educational prototype only. The bundled data is synthetic and is not suitable "
    "for production security decisions. Predictions are not proof of malicious activity."
)

@st.cache_resource
def get_model():
    return load_or_train()

model = get_model()
left, right = st.columns(2)
with left:
    st.subheader("Analyze a CSV file")
    st.write("Upload a CSV containing these features: " + ", ".join(FEATURES))
    uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = set(FEATURES) - set(incoming.columns)
            if missing:
                st.error("Missing required columns: " + ", ".join(sorted(missing)))
            else:
                predictions = model.predict(incoming[FEATURES])
                results = incoming.copy()
                results["prediction"] = predictions
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(incoming[FEATURES])
                    classes = list(model.classes_)
                    if "ATTACK" in classes:
                        results["attack_probability"] = probabilities[:, classes.index("ATTACK")].round(4)
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="intrusion_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Could not process this file: {exc}")

with right:
    st.subheader("Try a single flow")
    with st.form("single_flow"):
        duration = st.number_input("Duration (seconds)", min_value=0.0, value=1.2, step=0.1)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        dst_port = st.number_input("Destination port", min_value=0, max_value=65535, value=443)
        packets = st.number_input("Packets", min_value=1, value=24)
        bytes_total = st.number_input("Bytes", min_value=0, value=4200)
        syn_count = st.number_input("SYN count", min_value=0, value=1)
        failed_logins = st.number_input("Failed logins", min_value=0, value=0)
        connection_rate = st.number_input("Connection rate", min_value=0.0, value=3.0, step=0.5)
        inbound_packets = st.number_input("Inbound packets", min_value=0, value=12)
        outbound_packets = st.number_input("Outbound packets", min_value=0, value=12)
        submitted = st.form_submit_button("Analyze flow")
    if submitted:
        sample = pd.DataFrame([{
            "duration": duration, "protocol": protocol, "dst_port": dst_port,
            "packets": packets, "bytes": bytes_total, "syn_count": syn_count,
            "failed_logins": failed_logins, "connection_rate": connection_rate,
            "inbound_packets": inbound_packets, "outbound_packets": outbound_packets
        }])[FEATURES]
        prediction = model.predict(sample)[0]
        st.metric("Model prediction", prediction)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            classes = list(model.classes_)
            if "ATTACK" in classes:
                st.metric("Estimated attack probability", f"{probabilities[classes.index('ATTACK')]:.1%}")
        st.info("Treat this as a triage signal, not a definitive security verdict.")

st.divider()
st.subheader("Included sample data")
sample_data = pd.read_csv(DATA_PATH)
st.write(f"{len(sample_data)} synthetic example flows")
st.dataframe(sample_data.head(10), use_container_width=True)
