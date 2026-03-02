
# 🏦 Bank Term Deposit Propensity Engine

This is a production-grade Streamlit web application that predicts the likelihood of a bank customer subscribing to a term deposit. It uses a trained Random Forest model to analyze demographic, financial, and previous marketing campaign data.

## ✨ Features
* **🧑‍💼 Single Customer Prediction:** Input individual customer details via a sidebar to get a real-time propensity score, visualized with a gauge chart.
* **📊 Global Feature Importance:** See exactly which customer attributes are driving the model's predictions.
* **📂 Batch Processing:** Upload a CSV of thousands of uncategorized customers, score them all at once, and download a prioritized target list.
* **📈 Model Telemetry:** View the underlying machine learning metrics (Confusion Matrix & Classification Report) directly in the app.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/bank-propensity-engine.git](https://github.com/your-username/bank-propensity-engine.git)
cd bank-propensity-engine

```

### 2. Create a virtual environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the app

Ensure you have your dataset files (`train.csv` and `test.csv`) in the root directory, then run:

```bash
streamlit run app.py

```

## 📁 Project Structure

* `app.py`: The main Streamlit application script.
* `train.csv` / `test.csv`: The datasets used for training the model and evaluating performance. *(Note: Add these locally)*
* `requirements.txt`: Python dependencies required to run the project.

