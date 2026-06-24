#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import kurtosis, skew
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
st.title("Bearing Fault Classification Dashboard")
st.write("Upload a vibration signal CSV file.")
#upload file
uploaded_file=st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
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
        st.subheader("Signal Plot")
        st.line_chart(signal)
        window_size=1000
        results=[]
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
        #results printing
        results_df=pd.DataFrame(
        results,
        columns=["Window","KNN Prediction","Random Forest Prediction","Severity"]
        )
        st.subheader("Prediction Results")
        st.dataframe(results_df,use_container_width=True)
        #summary
        st.subheader("Fault Summary")
        st.write(results_df["KNN Prediction"].value_counts())
        