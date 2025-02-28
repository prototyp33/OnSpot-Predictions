#!/usr/bin/env python
"""
Enhanced dashboard for visualizing model performance, including cross-validation results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import glob
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys

# Add the project root to the Python path to allow imports from scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page config
st.set_page_config(
    page_title="Parking Occupancy Model Dashboard",
    page_icon="🅿️",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Model Explorer", "Cross-Validation Results", "Hyperparameter Tuning", "Feature Impact"]
)

# Function to load cross-validation results
def load_cv_results():
    cv_dir = "cross_validation_results"
    if not os.path.exists(cv_dir):
        return None
    
    # Look for summary file
    summary_path = os.path.join(cv_dir, "cv_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_text = f.read()
    else:
        summary_text = "No cross-validation summary found."
    
    # Look for visualization files
    vis_files = glob.glob(os.path.join(cv_dir, "*.png"))
    
    return {
        'summary_text': summary_text,
        'visualization_files': vis_files
    }

# Function to load hyperparameter tuning results
def load_tuning_results():
    tuning_dir = "hyperparameter_tuning_results"
    if not os.path.exists(tuning_dir):
        return None
    
    # Look for summary file
    summary_path = os.path.join(tuning_dir, "tuning_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_text = f.read()
    else:
        summary_text = "No hyperparameter tuning summary found."
    
    # Look for visualization files
    vis_dir = os.path.join(tuning_dir, "visualizations")
    if os.path.exists(vis_dir):
        vis_files = glob.glob(os.path.join(vis_dir, "*.png"))
    else:
        vis_files = []
    
    return {
        'summary_text': summary_text,
        'visualization_files': vis_files
    }

# Function to load feature impact results
def load_feature_impact_results():
    impact_dir = "feature_impact_results"
    if not os.path.exists(impact_dir):
        return None
    
    # Look for summary file
    summary_path = os.path.join(impact_dir, "model_comparison_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_text = f.read()
    else:
        summary_text = "No feature impact summary found."
    
    # Look for model files
    model_files = glob.glob(os.path.join(impact_dir, "*.pkl"))
    
    return {
        'summary_text': summary_text,
        'model_files': model_files
    }

# Model Explorer page
if page == "Model Explorer":
    st.title("Parking Occupancy Model Explorer")
    
    # Find all model files from different sources
    model_dirs = ["feature_impact_results", "trained_models", "hyperparameter_tuning_results/global_models", "hyperparameter_tuning_results/location_models"]
    model_files = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            model_files.extend(glob.glob(os.path.join(model_dir, "*.pkl")))
    
    if not model_files:
        st.warning("No model files found. Please train models first.")
    else:
        # Sidebar for model selection
        selected_model = st.sidebar.selectbox("Select Model", [os.path.basename(f) for f in model_files])
        model_path = [f for f in model_files if os.path.basename(f) == selected_model][0]
        
        # Load the selected model
        model = joblib.load(model_path)
        
        # Display model information
        st.header("Model Information")
        st.write(f"**Model Path:** {model_path}")
        
        if hasattr(model, 'named_steps'):
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
        else:
            st.write(f"**Model Type:** {type(model).__name__}")
            
            # Model parameters
            st.subheader("Model Parameters")
            params = model.get_params()
            st.json(params)

# Cross-Validation Results page
elif page == "Cross-Validation Results":
    st.title("Cross-Validation Results")
    
    cv_results = load_cv_results()
    
    if cv_results is None:
        st.warning("No cross-validation results found. Please run cross-validation first.")
    else:
        # Display summary
        st.header("Cross-Validation Summary")
        st.text(cv_results['summary_text'])
        
        # Display visualizations
        if cv_results['visualization_files']:
            st.header("Visualizations")
            
            # Group visualizations by type
            global_vis = [f for f in cv_results['visualization_files'] if 'global' in os.path.basename(f)]
            location_vis = [f for f in cv_results['visualization_files'] if 'location' in os.path.basename(f)]
            
            # Global visualizations
            if global_vis:
                st.subheader("Global Model Results")
                cols = st.columns(2)
                for i, vis_file in enumerate(global_vis):
                    with cols[i % 2]:
                        st.image(vis_file, caption=os.path.basename(vis_file), use_column_width=True)
            
            # Location visualizations
            if location_vis:
                st.subheader("Location-Specific Model Results")
                
                # Group by location
                location_ids = set()
                for vis_file in location_vis:
                    filename = os.path.basename(vis_file)
                    if 'location_' in filename:
                        location_id = filename.split('_')[1]
                        location_ids.add(location_id)
                
                # Create tabs for each location
                if location_ids:
                    tabs = st.tabs([f"Location {loc}" for loc in sorted(location_ids)])
                    
                    for i, loc in enumerate(sorted(location_ids)):
                        with tabs[i]:
                            loc_files = [f for f in location_vis if f'location_{loc}' in os.path.basename(f)]
                            cols = st.columns(2)
                            for j, vis_file in enumerate(loc_files):
                                with cols[j % 2]:
                                    st.image(vis_file, caption=os.path.basename(vis_file), use_column_width=True)
                
                # Location comparison visualizations
                comparison_vis = [f for f in location_vis if 'location_model_comparison' in os.path.basename(f) or 'location_r2_vs_data_size' in os.path.basename(f)]
                if comparison_vis:
                    st.subheader("Location Comparison")
                    for vis_file in comparison_vis:
                        st.image(vis_file, caption=os.path.basename(vis_file), use_column_width=True)
        else:
            st.info("No visualization files found.")

# Hyperparameter Tuning page
elif page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning Results")
    
    tuning_results = load_tuning_results()
    
    if tuning_results is None:
        st.warning("No hyperparameter tuning results found. Please run hyperparameter tuning first.")
    else:
        # Display summary
        st.header("Hyperparameter Tuning Summary")
        st.text(tuning_results['summary_text'])
        
        # Display visualizations
        if tuning_results['visualization_files']:
            st.header("Visualizations")
            
            # Group visualizations by model type and feature set
            model_types = ['gbm', 'rf']
            feature_sets = ['basic', 'advanced']
            
            # Create tabs for model types
            model_tabs = st.tabs([f"{model.upper()} Models" for model in model_types])
            
            for i, model_type in enumerate(model_types):
                with model_tabs[i]:
                    # Create subtabs for feature sets
                    feature_tabs = st.tabs([f"{feature.title()} Features" for feature in feature_sets])
                    
                    for j, feature_set in enumerate(feature_sets):
                        with feature_tabs[j]:
                            # Global model visualizations
                            global_vis = [f for f in tuning_results['visualization_files'] 
                                         if f'global_{model_type}_{feature_set}' in os.path.basename(f)]
                            
                            if global_vis:
                                st.subheader("Global Model")
                                cols = st.columns(2)
                                for k, vis_file in enumerate(global_vis):
                                    with cols[k % 2]:
                                        st.image(vis_file, caption=os.path.basename(vis_file), use_column_width=True)
                            
                            # Location-specific visualizations
                            location_vis = [f for f in tuning_results['visualization_files'] 
                                           if f'location_' in os.path.basename(f) and f'_{model_type}_{feature_set}' in os.path.basename(f)]
                            
                            if location_vis:
                                st.subheader("Location-Specific Models")
                                
                                # Group by location
                                location_ids = set()
                                for vis_file in location_vis:
                                    filename = os.path.basename(vis_file)
                                    if 'location_' in filename:
                                        location_id = filename.split('_')[1]
                                        location_ids.add(location_id)
                                
                                # Create expanders for each location
                                for loc in sorted(location_ids):
                                    with st.expander(f"Location {loc}"):
                                        loc_files = [f for f in location_vis if f'location_{loc}_{model_type}_{feature_set}' in os.path.basename(f)]
                                        cols = st.columns(2)
                                        for l, vis_file in enumerate(loc_files):
                                            with cols[l % 2]:
                                                st.image(vis_file, caption=os.path.basename(vis_file), use_column_width=True)
        else:
            st.info("No visualization files found.")

# Feature Impact page
elif page == "Feature Impact":
    st.title("Feature Impact Analysis")
    
    impact_results = load_feature_impact_results()
    
    if impact_results is None:
        st.warning("No feature impact results found. Please run feature impact analysis first.")
    else:
        # Display summary
        st.header("Feature Impact Summary")
        st.text(impact_results['summary_text'])
        
        # Display model comparison
        if impact_results['model_files']:
            st.header("Model Comparison")
            
            # Group models by type
            global_models = [f for f in impact_results['model_files'] if 'global_model' in os.path.basename(f)]
            location_models = [f for f in impact_results['model_files'] if 'location_' in os.path.basename(f)]
            
            # Create tabs for model types
            model_tabs = st.tabs(["Global Models", "Location-Specific Models"])
            
            with model_tabs[0]:
                # Group by feature set
                basic_models = [f for f in global_models if 'basic_features' in os.path.basename(f)]
                advanced_models = [f for f in global_models if 'advanced_features' in os.path.basename(f)]
                
                st.subheader("Basic Features")
                if basic_models:
                    for model_file in basic_models:
                        model = joblib.load(model_file)
                        st.write(f"**Model:** {os.path.basename(model_file)}")
                        
                        # Display feature importance if available
                        if hasattr(model.named_steps['model'], 'feature_importances_'):
                            # Get feature names
                            try:
                                preprocessor = model.named_steps['preprocessor']
                                feature_names = preprocessor.get_feature_names_out()
                            except:
                                feature_names = [f"Feature_{i}" for i in range(len(model.named_steps['model'].feature_importances_))]
                            
                            # Get feature importances
                            importances = model.named_steps['model'].feature_importances_
                            
                            # Sort and display top features
                            indices = np.argsort(importances)[::-1]
                            top_n = 20
                            
                            # Create DataFrame for display
                            importance_df = pd.DataFrame({
                                'Feature': [feature_names[i] for i in indices][:top_n],
                                'Importance': importances[indices][:top_n]
                            })
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                            ax.set_title("Feature Importances (Basic Features)")
                            st.pyplot(fig)
                else:
                    st.info("No global models with basic features found.")
                
                st.subheader("Advanced Features")
                if advanced_models:
                    for model_file in advanced_models:
                        model = joblib.load(model_file)
                        st.write(f"**Model:** {os.path.basename(model_file)}")
                        
                        # Display feature importance if available
                        if hasattr(model.named_steps['model'], 'feature_importances_'):
                            # Get feature names
                            try:
                                preprocessor = model.named_steps['preprocessor']
                                feature_names = preprocessor.get_feature_names_out()
                            except:
                                feature_names = [f"Feature_{i}" for i in range(len(model.named_steps['model'].feature_importances_))]
                            
                            # Get feature importances
                            importances = model.named_steps['model'].feature_importances_
                            
                            # Sort and display top features
                            indices = np.argsort(importances)[::-1]
                            top_n = 20
                            
                            # Create DataFrame for display
                            importance_df = pd.DataFrame({
                                'Feature': [feature_names[i] for i in indices][:top_n],
                                'Importance': importances[indices][:top_n]
                            })
                            
                            # Plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                            ax.set_title("Feature Importances (Advanced Features)")
                            st.pyplot(fig)
                else:
                    st.info("No global models with advanced features found.")
            
            with model_tabs[1]:
                if location_models:
                    # Group by location
                    location_ids = set()
                    for model_file in location_models:
                        filename = os.path.basename(model_file)
                        if 'location_' in filename:
                            location_id = filename.split('_')[1]
                            location_ids.add(location_id)
                    
                    # Create tabs for each location
                    if location_ids:
                        location_tabs = st.tabs([f"Location {loc}" for loc in sorted(location_ids)])
                        
                        for i, loc in enumerate(sorted(location_ids)):
                            with location_tabs[i]:
                                # Group by feature set
                                loc_basic_models = [f for f in location_models if f'location_{loc}' in os.path.basename(f) and 'basic_features' in os.path.basename(f)]
                                loc_advanced_models = [f for f in location_models if f'location_{loc}' in os.path.basename(f) and 'advanced_features' in os.path.basename(f)]
                                
                                st.subheader("Basic Features")
                                if loc_basic_models:
                                    for model_file in loc_basic_models:
                                        model = joblib.load(model_file)
                                        st.write(f"**Model:** {os.path.basename(model_file)}")
                                        
                                        # Display feature importance if available
                                        if hasattr(model.named_steps['model'], 'feature_importances_'):
                                            # Get feature names
                                            try:
                                                preprocessor = model.named_steps['preprocessor']
                                                feature_names = preprocessor.get_feature_names_out()
                                            except:
                                                feature_names = [f"Feature_{i}" for i in range(len(model.named_steps['model'].feature_importances_))]
                                            
                                            # Get feature importances
                                            importances = model.named_steps['model'].feature_importances_
                                            
                                            # Sort and display top features
                                            indices = np.argsort(importances)[::-1]
                                            top_n = 20
                                            
                                            # Create DataFrame for display
                                            importance_df = pd.DataFrame({
                                                'Feature': [feature_names[i] for i in indices][:top_n],
                                                'Importance': importances[indices][:top_n]
                                            })
                                            
                                            # Plot
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                                            ax.set_title(f"Feature Importances for Location {loc} (Basic Features)")
                                            st.pyplot(fig)
                                else:
                                    st.info(f"No models for Location {loc} with basic features found.")
                                
                                st.subheader("Advanced Features")
                                if loc_advanced_models:
                                    for model_file in loc_advanced_models:
                                        model = joblib.load(model_file)
                                        st.write(f"**Model:** {os.path.basename(model_file)}")
                                        
                                        # Display feature importance if available
                                        if hasattr(model.named_steps['model'], 'feature_importances_'):
                                            # Get feature names
                                            try:
                                                preprocessor = model.named_steps['preprocessor']
                                                feature_names = preprocessor.get_feature_names_out()
                                            except:
                                                feature_names = [f"Feature_{i}" for i in range(len(model.named_steps['model'].feature_importances_))]
                                            
                                            # Get feature importances
                                            importances = model.named_steps['model'].feature_importances_
                                            
                                            # Sort and display top features
                                            indices = np.argsort(importances)[::-1]
                                            top_n = 20
                                            
                                            # Create DataFrame for display
                                            importance_df = pd.DataFrame({
                                                'Feature': [feature_names[i] for i in indices][:top_n],
                                                'Importance': importances[indices][:top_n]
                                            })
                                            
                                            # Plot
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                                            ax.set_title(f"Feature Importances for Location {loc} (Advanced Features)")
                                            st.pyplot(fig)
                                else:
                                    st.info(f"No models for Location {loc} with advanced features found.")
                else:
                    st.info("No location-specific models found.")
        else:
            st.info("No model files found.") 