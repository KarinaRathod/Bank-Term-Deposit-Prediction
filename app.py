import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Propensity Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, clean look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- DATA & MODEL PIPELINE ----------------
@st.cache_resource
def get_pipeline(df):
    """Trains and caches the model pipeline."""
    X = df.drop("y", axis=1)
    y = df["y"].map({"yes": 1, "no": 0})
    
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # Using class_weight='balanced' to handle the typical 90/10 split in bank data
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42))
    ])
    
    return clf.fit(X, y), cat_cols, num_cols

@st.cache_data
def load_data():
    try:
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        return train, test
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please ensure `train.csv` and `test.csv` are in the directory.")
        return None, None

# ---------------- INITIALIZATION ----------------
train_df, test_df = load_data()

if train_df is not None:
    model, cat_cols, num_cols = get_pipeline(train_df)

    # ---------------- SIDEBAR: INPUTS ----------------
    st.sidebar.title("⚙️ Customer Inputs")
    st.sidebar.markdown("Adjust parameters below for **Single Prediction**.")
    
    def get_user_inputs():
        inputs = {}
        with st.sidebar.expander("👤 Demographics", expanded=True):
            inputs['age'] = st.slider("Age", 18, 95, 35)
            inputs['job'] = st.selectbox("Job", train_df['job'].unique())
            inputs['marital'] = st.selectbox("Marital Status", train_df['marital'].unique())
            inputs['education'] = st.selectbox("Education", train_df['education'].unique())

        with st.sidebar.expander("💰 Financial Status", expanded=True):
            inputs['balance'] = st.number_input("Average Yearly Balance (€)", value=1500)
            inputs['housing'] = st.selectbox("Has Housing Loan?", train_df['housing'].unique())
            inputs['loan'] = st.selectbox("Has Personal Loan?", train_df['loan'].unique())
            inputs['default'] = st.selectbox("Has Credit in Default?", train_df['default'].unique())

        with st.sidebar.expander("📞 Campaign Details", expanded=False):
            inputs['contact'] = st.selectbox("Contact Communication", train_df['contact'].unique())
            
            # --> THE FIX: Added the 'day' column input <--
            inputs['day'] = st.slider("Last Contact Day of Month", int(train_df['day'].min()), int(train_df['day'].max()), 15)
            
            inputs['month'] = st.select_slider("Last Contact Month", options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            inputs['duration'] = st.number_input("Last Contact Duration (sec)", value=200)
            inputs['campaign'] = st.number_input("Contacts during Campaign", min_value=1, value=1)
            inputs['pdays'] = st.number_input("Days since last contact (-1 for never)", value=-1)
            inputs['previous'] = st.number_input("Previous contacts", value=0)
            inputs['poutcome'] = st.selectbox("Outcome of previous campaign", train_df['poutcome'].unique())
            
        return pd.DataFrame([inputs])

    input_df = get_user_inputs()

    # ---------------- MAIN APP LAYOUT ----------------
    st.title("🏦 Term Deposit Propensity Engine")
    st.markdown("Predict which customers are most likely to subscribe to a term deposit based on historical campaign data.")
    
    tab1, tab2, tab3 = st.tabs(["🧑‍💼 Single Prediction", "📂 Batch Processing", "📈 Model Telemetry"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Targeting Recommendation")
            
            prob = model.predict_proba(input_df)[0][1]
            pred = 1 if prob > 0.5 else 0

            # Beautiful Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%", 'valueformat': ".1f"},
                title = {'text': "Propensity to Subscribe"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps' : [
                        {'range': [0, 33], 'color': "#ffdddd"},
                        {'range': [33, 66], 'color': "#ffffcc"},
                        {'range': [66, 100], 'color': "#ddffdd"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if pred == 1:
                st.success("#### ✅ High Priority Lead\nThis customer matches the profile of historical subscribers. Recommend immediate outreach.")
            else:
                st.warning("#### ⏸️ Low Priority Lead\nThis customer is unlikely to convert at this time. Save resources for higher propensity targets.")

        with col2:
            st.subheader("Global Feature Importance")
            importances = model.named_steps['classifier'].feature_importances_
            ohe_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
            feature_names = np.concatenate([num_cols, ohe_features])
            
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(8)
            
            fig_bar = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', 
                         color='Importance', color_continuous_scale='Blues')
            fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 2: BATCH PROCESSING ---
    with tab2:
        st.subheader("Batch Prediction Pipeline")
        st.markdown("Upload a CSV of uncategorized customers to generate propensity scores in bulk.")
        
        # We expect a CSV with the same columns as train.csv (minus 'y')
        uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(batch_df)} records. Processing...")
            
            try:
                # Predict
                batch_probs = model.predict_proba(batch_df)[:, 1]
                batch_preds = (batch_probs > 0.5).astype(int)
                
                # Append results
                results_df = batch_df.copy()
                results_df.insert(0, 'Recommendation', np.where(batch_preds == 1, 'Contact', 'Skip'))
                results_df.insert(0, 'Propensity_Score', np.round(batch_probs * 100, 2))
                
                # Sort by highest propensity
                results_df = results_df.sort_values(by='Propensity_Score', ascending=False)
                
                st.success("Batch processing complete!")
                st.dataframe(results_df.head(100), use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Leads",
                    data=csv,
                    file_name="scored_customers.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error processing batch. Please ensure column names match training data. Details: {e}")

    # --- TAB 3: MODEL TELEMETRY ---
    with tab3:
        st.subheader("Model Performance on Test Set")
        
        y_test_pred = model.predict(test_df.drop('y', axis=1))
        y_test_true = test_df['y'].map({"yes": 1, "no": 0})
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test_true, y_test_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                               x=['No Subscription', 'Subscription'], y=['No Subscription', 'Subscription'])
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("**Classification Report**")
            report = classification_report(y_test_true, y_test_pred, output_dict=True, target_names=['No (0)', 'Yes (1)'])
            report_df = pd.DataFrame(report).transpose().round(2)
            
            st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['f1-score']), use_container_width=True)