# scripts/model_dashboard.py
import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import glob

# Set page title
st.title("Parking Occupancy Model Inspector")

# Find all model files
model_dir = "feature_impact_results"
model_files = glob.glob(os.path.join(model_dir, "*.pkl"))
model_names = [os.path.basename(f) for f in model_files]

# Sidebar for model selection
selected_model = st.sidebar.selectbox("Select Model", model_names)
model_path = os.path.join(model_dir, selected_model)

# Load the selected model
model = joblib.load(model_path)

# Display model information
st.header("Model Information")
st.write(f"**Model Type:** {type(model).__name__}")
st.write(f"**Pipeline Steps:** {list(model.named_steps.keys())}")

# Get the actual model (not the pipeline)
estimator = model.named_steps['model']
st.write(f"**Estimator Type:** {type(estimator).__name__}")

# Model parameters
st.subheader("Model Parameters")
params = estimator.get_params()
st.json(params)

# Feature importance
if hasattr(estimator, 'feature_importances_'):
    st.header("Feature Importance")
    
    # Get feature names from the pipeline
    try:
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"Feature_{i}" for i in range(len(estimator.feature_importances_))]
    
    # Get feature importances
    importances = estimator.feature_importances_
    
    # Sort and display top features
    indices = np.argsort(importances)[::-1]
    top_n = st.slider("Number of top features to display", 5, 50, 20)
    
    # Create DataFrame for display
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices][:top_n],
        'Importance': importances[indices][:top_n]
    })
    
    # Display as table
    st.dataframe(importance_df)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)

# Model performance metrics (if available)
st.header("Model Performance")
summary_path = os.path.join(model_dir, "model_comparison_summary.txt")
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary = f.read()
    st.text(summary)