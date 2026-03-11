# RC-Coupling-Beam-Vu-Predictor

## Interpretable Machine-Learning Tool for Predicting Ultimate Shear Capacity of RC Coupling Beams

This repository provides a lightweight **Streamlit-based graphical user interface (GUI)** and Python implementation for predicting the **ultimate shear capacity ($V_u$)** of reinforced concrete (RC) coupling beams using machine-learning models.

The tool allows engineers and researchers to estimate the shear capacity by simply entering key geometric, material, and reinforcement parameters of RC coupling beams.

---

## 🔍 Features

- Predict **ultimate shear capacity ($V_u$)** of RC coupling beams
- Multiple machine-learning models implemented:
  - Random Forest
  - XGBoost
  - CatBoost
  - LightGBM
  - Multi-Layer Perceptron (MLP)
  - Practical Solution (PS)
- User-friendly **Streamlit GUI**
- Instant prediction results
- Export prediction results as **CSV**
- Supports both **conventionally reinforced** and **diagonally reinforced coupling beams**

---

## 🖥 Application Interface

The GUI allows users to input the following parameters:

### Geometric properties
- Beam length ($l$)
- Beam height ($h$)
- Beam width ($b$)

### Material properties
- Concrete compressive strength ($f'_c$)
- Yield strength of longitudinal reinforcement ($f_{yl}$)
- Yield strength of web stirrups ($f_{yv}$)
- Yield strength of diagonal reinforcement ($f_{yd}$)

### Reinforcement properties
- Longitudinal reinforcement ratio ($\rho_l$)
- Web stirrup reinforcement ratio ($\rho_v$)
- Web Stirrup spacing ($s$)
- Diagonal reinforcement ratio ($\rho_d$)
- Diagonal bar angle ($\alpha$)

---

## 🚀 Running the App Locally

Clone the repository:

```bash
git clone https://github.com/your-repository-name.git
cd RC-Coupling-Beam-Vu-Predictor
