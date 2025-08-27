# Lag Selection Visualizer

A Python tool to **visualize and compare lag order selection** for multivariate time series models using **AIC** (Akaike Information Criterion) and **BIC** (Bayesian Information Criterion).  
It produces both tabular results and an **animated GIF** showing the evolution of AIC/BIC values across lags.

---

## 🚀 Features
- Computes AIC and BIC values for different lag orders in **VAR (Vector Autoregression)** models.
- Highlights the **best lag order** based on each criterion.
- Generates an **animated visualization** (`.gif`) of the selection process.
- Object-oriented design for easy extension and integration.

---

## 📦 Installation
Clone this repository and install required dependencies:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
