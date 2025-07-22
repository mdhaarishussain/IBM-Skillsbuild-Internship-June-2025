import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary Prediction | Md Haaris Hussain",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Preprocessors ---
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessors from disk."""
    model = joblib.load("best_model.pkl")
    try:
        preprocessors = joblib.load("preprocessors.pkl")
    except FileNotFoundError:
        # If preprocessors.pkl is not found, create default ones
        preprocessors = create_default_preprocessors()
    return model, preprocessors

def create_default_preprocessors():
    """Create default label encoders based on the Adult dataset's known categories."""
    preprocessors = {}
    categories = {
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
        'gender': ['Male', 'Female'],
        'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    }
    for column, cats in categories.items():
        le = LabelEncoder()
        le.fit(cats)
        preprocessors[column] = le
    return preprocessors

model, preprocessors = load_model_and_preprocessors()

# --- Custom CSS for Modern DARK MODE UI ---
st.markdown("""
<style>
    /* General Styles */
    body {
        color: #EAEAEA;
    }
    .stApp {
        background-color: #0E1117; /* Main Dark Background */
    }

    /* Main Title & Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
        color: #3B82F6; /* Bright Blue */
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #A0AEC0; /* Light Gray */
        margin-bottom: 25px;
    }
    .author-line {
        font-size: 1rem;
        text-align: center;
        color: #3B82F6;
        font-weight: bold;
        margin-bottom: 30px;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1A202C; /* Darker Sidebar */
        border-right: 1px solid #2D3748;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    .stSidebar .st-emotion-cache-16txtl3 h2, .stSidebar .st-emotion-cache-16txtl3 h3 {
       color: #3B82F6;
       font-weight: bold;
    }
    .stSidebar .st-emotion-cache-16txtl3, .stSidebar .st-emotion-cache-16txtl3 p, .stSidebar .st-emotion-cache-16txtl3 label {
        color: #EAEAEA;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #1A202C; /* Dark Card Background */
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #2D3748; /* Dark Border */
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #3B82F6;
    }
    .metric-card h3 {
        font-size: 2rem;
        font-weight: bold;
        color: #3B82F6;
        margin: 0;
    }
    .metric-card p {
        color: #A0AEC0;
        font-size: 1rem;
        margin: 5px 0 0 0;
    }
    
    /* Prediction Box */
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        font-size: 1.3rem;
        font-weight: bold;
        border: 2px solid;
    }
    .high-salary {
        background-color: #1F2937; /* Dark Greenish-Gray */
        color: #34D399; /* Bright Green */
        border-color: #10B981;
    }
    .low-salary {
        background-color: #1F2937; /* Dark Reddish-Gray */
        color: #F87171; /* Bright Red */
        border-color: #EF4444;
    }

    /* Buttons */
    .stButton > button {
        background-color: #3B82F6; /* Bright Blue */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: bold;
        width: 100%;
        transition: background-color 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.25);
    }
    .stButton > button:hover {
        background-color: #2563EB; /* Darker Blue */
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35);
    }
    
    /* Internship & Socials */
    .internship-badge {
        text-align: center;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 15px;
        border-radius: 12px;
        margin: auto;
        margin-bottom: 30px;
        font-weight: bold;
        width: 80%;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #A0AEC0;
        font-size: 0.9rem;
    }

    /* Other elements */
    hr {
        background-color: #2D3748;
    }
    .st-emotion-cache-1r6slb0, .st-emotion-cache-1hdb6h2 { /* Headers in main body */
        color: #EAEAEA;
    }
    .st-emotion-cache-q8sbsg { /* Expander header */
        color: #EAEAEA;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def preprocess_input(input_df):
    """Apply label encoding to the input dataframe."""
    processed_df = input_df.copy()
    categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    
    for col in categorical_columns:
        if col in processed_df.columns and col in preprocessors:
            try:
                # Transform data using the loaded preprocessor
                processed_df[col] = preprocessors[col].transform(processed_df[col])
            except ValueError:
                # Handle unknown categories by assigning a default value (e.g., 0)
                st.warning(f"An unknown category was found in '{col}'. Using a default value.")
                processed_df[col] = 0
    return processed_df

def create_input_dataframe(age, workclass, fnlwgt, educational_num, marital_status, 
                          occupation, relationship, race, gender, capital_gain, capital_loss, 
                          hours_per_week, native_country):
    """Create a DataFrame from user inputs with correct column names."""
    return pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'educational-num': [educational_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.image("https://edunetfoundation.org/wp-content/uploads/2022/06/edunet-logo-white.png", width=200)
    st.image("https://www.gesi.org/wp-content/uploads/2024/08/purepng.com-ibm-logologobrand-logoiconslogos-251519939176ka7y8.png", width=200)
    st.markdown("## 🔧 Employee Details")
    st.markdown("Enter the employee's information to predict their salary class.")
    
    # --- Input Sections ---
    st.markdown("### 👤 Demographics")
    age = st.slider("Age", 18, 90, 39, help="Employee's age in years.")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Employee's gender.")
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], help="Employee's race.")
    
    st.markdown("### 🎓 Education")
    education_map = {
        "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
        "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
        "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
        "Prof-school": 15, "Doctorate": 16
    }
    education_display = st.selectbox("Education Level", list(education_map.keys()), index=12, help="Highest education level achieved.")
    educational_num = education_map[education_display]

    st.markdown("### 💼 Work Information")
    workclass = st.selectbox("Work Class", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"], help="Type of employment.")
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'], help="Current occupation category.")
    hours_per_week = st.slider("Hours per Week", 1, 99, 40, help="Average working hours per week.")
    
    st.markdown("### 💰 Financial & Personal")
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"], help="Current marital status.")
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"], help="Relationship status in household.")
    native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'], help="Country of origin.")
    fnlwgt = st.number_input("Final Weight", 12285, 1484705, 189778, help="Census weight (sampling weight).")
    capital_gain = st.number_input("Capital Gain", 0, 99999, 0, help="Capital gains income.")
    capital_loss = st.number_input("Capital Loss", 0, 4356, 0, help="Capital losses.")

# --- Main App Content ---
st.markdown('<h1 class="main-header">💼 Employee Salary Classification | Capstone Project</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An Advanced ML model to predict whether an employee earns >$50K or ≤$50K.</p>', unsafe_allow_html=True)
st.markdown('<p class="author-line">A Project by Md Haaris Hussain</p>', unsafe_allow_html=True)

# --- Internship Badge ---
st.markdown("""
<div class="internship-badge">
    🎓 EDUNET FOUNDATION - IBM SKILLSBUILD AI INTERNSHIP (JUNE 2025)
</div>
""", unsafe_allow_html=True)


# --- Model Performance Metrics ---
st.markdown("### 📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3>85.71%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>85%</h3><p>Precision</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>86%</h3><p>Recall</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3>85%</h3><p>F1-Score</p></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# --- Prediction Section ---
st.markdown("### 🎯 Salary Prediction")

# Create input dataframe
input_df = create_input_dataframe(
    age, workclass, fnlwgt, educational_num, marital_status, occupation,
    relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country
)

# Display a summary of the inputs
with st.expander("🔍 Review Your Selections", expanded=False):
    summary_data = {
        "Feature": ["Age", "Gender", "Education", "Occupation", "Hours/Week", "Marital Status"],
        "Value": [age, gender, education_display, occupation, f"{hours_per_week} hrs", marital_status]
    }
    st.table(pd.DataFrame(summary_data))
    st.markdown("##### Full Model Input:")
    st.dataframe(input_df, use_container_width=True)

# Predict button and result
if st.button("🚀 Predict Salary Class", key="predict_btn"):
    try:
        with st.spinner("🤖 The AI is analyzing the profile..."):
            processed_input = preprocess_input(input_df)
            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0]
            confidence = max(probability) * 100

            if prediction == ">50K":
                st.markdown(f'<div class="prediction-box high-salary">🎉 High Salary Prediction: <strong>{prediction}</strong><br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box low-salary">📉 Low Salary Prediction: <strong>{prediction}</strong><br>Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
            
            # Show probability distribution
            st.markdown("#### Prediction Confidence Distribution")
            prob_data = pd.DataFrame({'Salary Class': ['≤$50K', '>$50K'], 'Probability': probability})
            st.bar_chart(prob_data.set_index('Salary Class'))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all input fields are correctly filled.")

# --- Batch Prediction Section ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 📂 Batch Prediction")
st.markdown("Upload a CSV file to predict salaries for multiple employees at once.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="The CSV must contain these columns: age, workclass, fnlwgt, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country"
)

if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.markdown("#### Data Preview")
        st.dataframe(batch_data.head(), use_container_width=True)

        required_columns = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        
        # Handle 'sex' vs 'gender' column name
        if 'sex' in batch_data.columns and 'gender' not in batch_data.columns:
            st.info("Detected 'sex' column. Renaming it to 'gender' for compatibility.")
            batch_data.rename(columns={'sex': 'gender'}, inplace=True)

        missing_cols = [col for col in required_columns if col not in batch_data.columns]
        if 'education' in batch_data.columns:
            st.warning("The 'education' column was found and will be ignored, as the model uses 'educational-num'.")
            batch_data.drop(columns=['education'], inplace=True)
        
        if missing_cols:
            st.error(f"Your file is missing the following required columns: `{', '.join(missing_cols)}`")
        else:
            if st.button("🔮 Generate Batch Predictions"):
                with st.spinner("Processing batch file..."):
                    try:
                        processed_batch = preprocess_input(batch_data[required_columns])
                        batch_preds = model.predict(processed_batch)
                        batch_proba = model.predict_proba(processed_batch)
                        
                        results_df = batch_data.copy()
                        results_df['Predicted_Salary_Class'] = batch_preds
                        results_df['Confidence'] = [f"{max(p)*100:.1f}%" for p in batch_proba]

                        st.markdown("#### ✅ Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name='salary_predictions.csv',
                            mime='text/csv'
                        )
                    except Exception as e:
                        st.error(f"Error during batch prediction: {e}")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>Developed by <strong>Md Haaris Hussain</strong></p>
    <p>Powered by IBM SkillsBuild & Edunet Foundation | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)