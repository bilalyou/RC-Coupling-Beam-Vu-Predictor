# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 2026
@author: Administrator
"""

import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb
import numpy as np
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# 1) CHECK WHETHER SCRIPT IS RUNNING THROUGH STREAMLIT
# =========================================================
def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


IS_STREAMLIT = is_running_in_streamlit()


# =========================================================
# 2) PAGE CONFIG ONLY WHEN RUNNING IN STREAMLIT
# =========================================================
if IS_STREAMLIT:
    st.set_page_config(page_title="Vu Estimator", layout="wide", page_icon="🧱")


# =========================================================
# 3) HELPER FUNCTIONS
# =========================================================
def normalize_input(x_raw, scaler):
    return scaler.transform(x_raw)


def denormalize_output(y_scaled, scaler):
    y_scaled = np.array([[y_scaled]])
    return scaler.inverse_transform(y_scaled)[0][0]


# =========================================================
# 4) LOAD MODELS
# =========================================================
if IS_STREAMLIT:

    def load_models():

        ann_ps_model = joblib.load(BASE_DIR / "ANN_PS_Model.joblib")
        ann_ps_scaler_X = joblib.load(BASE_DIR / "ANN_PS_Scaler_X.save")
        ann_ps_scaler_y = joblib.load(BASE_DIR / "ANN_PS_Scaler_y.save")

        ann_mlp_model = joblib.load(BASE_DIR / "ANN_MLP_Model.joblib")
        ann_mlp_scaler_X = joblib.load(BASE_DIR / "ANN_MLP_Scaler_X.save")
        ann_mlp_scaler_y = joblib.load(BASE_DIR / "ANN_MLP_Scaler_y.save")

        rf_model = joblib.load(BASE_DIR / "Best_RF_Model_GUI.joblib")

        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(BASE_DIR / "Best_XGBoost_Model.json"))

        cat_model = catboost.CatBoostRegressor()
        cat_model.load_model(str(BASE_DIR / "Best_CatBoost_Model.cbm"))

        lgb_model = lgb.Booster(model_file=str(BASE_DIR / "Best_LightGBM_Model.txt"))

        models = {
            "XGBoost": xgb_model,
            "CatBoost": cat_model,
            "LightGBM": lgb_model,
            "PS": ann_ps_model,
            "MLP": ann_mlp_model,
            "Random Forest": rf_model
        }

        scalers = {
            "PS_X": ann_ps_scaler_X,
            "PS_y": ann_ps_scaler_y,
            "MLP_X": ann_mlp_scaler_X,
            "MLP_y": ann_mlp_scaler_y
        }

        return models, scalers

else:

    def load_models():
        return None, None


# =========================================================
# 5) GUI
# =========================================================
if IS_STREAMLIT:

    models, scalers = load_models()

    st.title("Ultimate Shear Capacity Estimator for RC Coupling Beams")

    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    col1, col2, col3 = st.columns(3)

    with col1:
        L = st.number_input("Beam Length l (mm)", value=1000.0)
        h = st.number_input("Beam Height h (mm)", value=400.0)
        b = st.number_input("Beam Width b (mm)", value=200.0)

    with col2:
        fc = st.number_input("Concrete Strength fc (MPa)", value=54.0)
        fyl = st.number_input("Yield Strength Longitudinal Bars fyl (MPa)", value=476.0)
        fyv = st.number_input("Yield Strength Stirrups fyv (MPa)", value=331.0)
        fyd = st.number_input("Yield Strength Diagonal Bars fyd (MPa)", value=476.0)

    with col3:
        Pl = st.number_input("Longitudinal Reinforcement Pl (%)", value=0.25)
        Pv = st.number_input("Stirrups Reinforcement Pv (%)", value=0.21)
        s = st.number_input("Stirrup Spacing s (mm)", value=150.0)
        Pd = st.number_input("Diagonal Reinforcement Pd (%)", value=1.005)
        alpha = st.number_input("Diagonal Angle alpha (deg)", value=17.5)

    model_choice = st.selectbox("Model Selection", list(models.keys()))

    submit = st.button("Calculate")

    if submit:

        input_array = np.array([[L, h, b, fc, fyl, fyv, Pl, Pv, s, Pd, fyd, alpha]])

        model = models[model_choice]

        if model_choice == "LightGBM":
            pred = model.predict(input_array)[0]

        elif model_choice == "PS":
            input_norm = normalize_input(input_array, scalers["PS_X"])
            pred_scaled = model.predict(input_norm)[0]
            pred = denormalize_output(pred_scaled, scalers["PS_y"])

        elif model_choice == "MLP":
            input_norm = normalize_input(input_array, scalers["MLP_X"])
            pred_scaled = model.predict(input_norm)[0]
            pred = denormalize_output(pred_scaled, scalers["MLP_y"])

        elif model_choice == "XGBoost":
            pred = model.predict(input_array)[0]

        else:
            pred = model.predict(input_array)[0]

        pred = max(pred, 0)

        st.success(f"Predicted Shear Capacity: {pred:.2f} kN")

        row = pd.DataFrame([[L,h,b,fc,fyl,fyv,Pl,Pv,s,Pd,fyd,alpha,pred]],
                           columns=["L","h","b","fc","fyl","fyv","Pl","Pv","s","Pd","fyd","alpha","Predicted_V"])

        st.session_state.results_df = pd.concat([st.session_state.results_df,row],ignore_index=True)

    if not st.session_state.results_df.empty:

        st.write("Recent Predictions")
        st.dataframe(st.session_state.results_df.tail())

        csv = st.session_state.results_df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            data=csv,
            file_name="shear_predictions.csv",
            mime="text/csv"
        )

    # --- Footer ---
    st.markdown("""
    <hr style='margin-top: 2rem;'>
    <div style='text-align: center; color: #888; font-size: 14px;'>
        Developed by [Your Name]. For academic and research purposes only.
    </div>

    """, unsafe_allow_html=True)


