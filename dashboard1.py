#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import kurtosis, skew\

#load models
knn_model=joblib.load("knn_model.pkl")
rf_model=joblib.load("random_forest_model.pkl")
severity_model=joblib.load("severity_model.pkl")
scaler=joblib.load("scaler.pkl")

#page settings
st.set_page_config(
    page_title="Bearing Fault Classification",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#00C8FF;
    margin-bottom:0px;
}

.sub-title{
    text-align:center;
    color:#B8BCC8;
    margin-bottom:30px;
}

.metric-card{
    background:#161B22;
    padding:15px;
    border-radius:15px;
    border:1px solid #30363D;
    text-align:center;
}

.result-card{
    background:#161B22;
    padding:20px;
    border-radius:15px;
    border-left:5px solid #00C8FF;
    margin-top:10px;
}

.footer{
    text-align:center;
    color:#808080;
    margin-top:50px;
}

.window-card{
    background:#161B22;
    padding:20px;
    border-radius:15px;
    border:1px solid #30363D;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class='main-title'>
Bearing Fault Classification System
</div>

<div class='sub-title'>
Machine Learning Based Vibration Signal Analysis Dashboard
</div>
""",
unsafe_allow_html=True
)

col1,col2=st.columns([3,1])

with col1:
    uploaded_file=st.file_uploader(
        "Upload Vibration Signal CSV",
        type=["csv"]
    )

with col2:
    st.info("""
Supported Faults
• Healthy
• Ball Fault
• Inner Race Fault
• Outer Race Fault
""")

#feature names
feature_names=["mean","std","rms","max","min","kurtosis","skewness"]

#preprocess the uploaded file
if uploaded_file is not None:
    test_signal=pd.read_csv(uploaded_file)
    if "signal" not in test_signal.columns:
        st.error("CSV file must contain a 'signal' column.")
    else:
        signal=test_signal["signal"].values
        
        st.subheader("Signal Preview")
        st.dataframe(test_signal.head())
        
        st.subheader("Signal Visualization")
        st.line_chart(pd.DataFrame({"Amplitude":signal}))
        
        window_size=1000
        results = []
        window_details = []
        
        for i in range(0,len(signal),window_size):
            window=signal[i:i+window_size]
            if len(window)<window_size:
                continue
                
            #feature extraction
            mean=np.mean(window)
            std=np.std(window)
            rms=np.sqrt(np.mean(window**2))
            maxi=np.max(window)
            mini=np.min(window)
            kurt=kurtosis(window)
            sk=skew(window)
            
            new_sample=pd.DataFrame(
            [[mean,std,rms,maxi,mini,kurt,sk]],
            columns=feature_names
            )
            
            #feature scaling
            new_sample_scaled=pd.DataFrame(
            scaler.transform(new_sample),
            columns=feature_names
            )
            
            #feature prediction
            knn_fault=knn_model.predict(new_sample_scaled)[0]
            rf_fault=rf_model.predict(new_sample_scaled)[0]
            severity=severity_model.predict(new_sample_scaled)[0]
            results.append([
            f"Window {len(results)+1}",
            knn_fault,
            rf_fault,
            severity
            ])
        
            window_details.append({
                "Window": f"Window {len(results)}",
                "Mean": mean,
                "STD": std,
                "RMS": rms,
                "Maximum": maxi,
                "Minimum": mini,
                "Kurtosis": kurt,
                "Skewness": sk,
                "Fault": knn_fault,
                "Severity": severity
            })
            
        #results printing
        results_df=pd.DataFrame(
        results,
        columns=["Window","KNN Prediction","Random Forest Prediction","Severity"]
        )
        
        st.subheader("Prediction Results")
        st.dataframe(results_df,use_container_width=True,height=400)

        st.subheader("Window Analysis")

        selected_window=st.selectbox(
            "Select a Window",
            options=[item["Window"] for item in window_details],
            key="window_selector"
        )
        
        selected_data=next(
            item for item in window_details
            if item["Window"] == selected_window
        )
        with st.container(border=True):
            st.markdown(f"## {selected_data['Window']}")
            col1,col2,col3,col4=st.columns(4)
            col1.metric("Mean", f"{selected_data['Mean']:.4f}")
            col2.metric("STD", f"{selected_data['STD']:.4f}")
            col3.metric("RMS", f"{selected_data['RMS']:.4f}")
            col4.metric("Maximum", f"{selected_data['Maximum']:.4f}")
            col5,col6,col7=st.columns(3)
            col5.metric("Minimum", f"{selected_data['Minimum']:.4f}")
            col6.metric("Kurtosis", f"{selected_data['Kurtosis']:.4f}")
            col7.metric("Skewness", f"{selected_data['Skewness']:.4f}")
            st.markdown("---")
            colA,colB=st.columns(2)
            with colA:
                st.markdown(f"**Predicted Fault:** {selected_data['Fault']}")
            with colB:
                st.markdown(f"**Severity:** {selected_data['Severity']}")
            severity=str(selected_data["Severity"]).lower()
            st.markdown("### Recommendation")
            if severity=="high":
                st.error("Immediate action required. Bearing inspection or replacement is recommended.")
            elif severity=="medium":
                st.warning("Maintenance should be scheduled soon.")
            else:
                st.success("No immediate action required. Continue monitoring.")
        st.subheader("Fault Distribution")
        fault_counts=results_df["KNN Prediction"].value_counts()
        st.bar_chart(fault_counts)
        st.markdown("---")
        st.markdown(
        """
        <div class='footer'>
        Bearing Fault Classification Dashboard<br>
        </div>
        """,
        unsafe_allow_html=True
        )
