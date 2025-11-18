import streamlit as st
import numpy as np
import pickle
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle
import statsmodels.api as sm
import requests
import os

# Things to do and add for this app
#1. Import the real model
#2. Set up all of the real features # <--- basically done
    # create a dataframe that contains all of the feature infomation such as the average, min, max, mode for categorical and etc
#3. Create an expected value calculation for letting someone through if they are fraud
#4. Allow for the option to enter the 'cost' of false negatives or maybe provide a graph instead
#5. Remember all of this would need to be ready by Thursday morning
#6. Place a QR code in the app somewhere? Or make it accessible by QR code?
#7. Do research regarding what needs to happen for this to be deployed so it can be accessed on their devices
#8. Organize the variables and the overall page to look like less of a shit show

# Remove days since requesst from the final model
# Fix the Jan, January situation
# FOr these categorical ones make sure to remove the option to even select them from the model


# Page configuration
st.set_page_config(page_title="Logistic Regression Predictor", layout="wide")

@st.cache_resource
def download_model_from_gdrive(file_id, destination):
    """Download file from Google Drive"""
    URL = "https://drive.google.com/file/d/1t1pEtXXkNQwxvZynEW4O7usNpmm6k51r/view?usp=sharing"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large files that require confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
    
    # Save the file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model():
    """Load the model, downloading from Google Drive if necessary"""
    model_path = "analytics_challenge_model_test.pickle"
    
    # If model doesn't exist locally, download it
    if not os.path.exists(model_path):
        file_id = "1t1pEtXXkNQwxvZynEW4O7usNpmm6k51r"  # Replace with your actual file ID
        with st.spinner("Downloading model from Google Drive..."):
            download_model_from_gdrive(file_id, model_path)
    
    return load_pickle(model_path)

# Load the model
model = load_model()
loaded_model = model

def prepare_model_input(input_values, all_features):
    """
    Convert user inputs into a properly formatted DataFrame with dummy variables
    that matches what the model expects
    """
    # Create a dictionary with the raw values
    data_dict = {}
    
    # Add numerical features directly as float
    for feature in all_features:
        if feature['type'] == 'number':
            data_dict[feature['name']] = [float(input_values[feature['name']])]
        elif feature['type'] == 'boolean':
            data_dict[feature['name']] = [int(input_values[feature['name']])]
    
    # Add categorical features (not yet encoded)
    categorical_features = {}
    for feature in all_features:
        if feature['type'] == 'categorical':
            categorical_features[feature['name']] = str(input_values[feature['name']])
    
    # Create initial DataFrame
    df = pd.DataFrame(data_dict)
    
    # Add categorical columns
    for col_name, value in categorical_features.items():
        df[col_name] = value
    
    # Now create dummy variables for categorical columns
    categorical_cols = [f['name'] for f in all_features if f['type'] == 'categorical']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Convert all columns to numeric
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    # The model expects specific columns in a specific order
    # Get the feature names from your trained model
    model_features = loaded_model.model.exog_names[1:]  # Skip intercept
    
    # Create a dataframe with all model features, filling missing ones with 0
    final_df = pd.DataFrame(0.0, index=[0], columns=model_features, dtype=float)
    
    # Fill in the values we have
    for col in df_encoded.columns:
        if col in final_df.columns:
            final_df[col] = float(df_encoded[col].values[0])
    
    return final_df

@st.cache_resource
def load_model():
    """Load the actual statsmodels GLM model"""
    return load_pickle("analytics_challenge_model_test.pickle")

# Load the actual model
model = load_model()
loaded_model = model  # Keep this for consistency with your existing code

# Define all available features
ALL_FEATURES = [
    {'name': 'income', 'label': 'Income', 'type': 'number', 'min': 0.0, 'max': 1.0, 'default': 0.5},
    {'name': 'name_email_similarity', 'label': 'Name Email Similarity', 'type': 'number', 'min': 0.0, 'max': 1.0, 'default': 0.5},
    {'name': 'salary', 'label': 'Salary', 'type': 'number', 'min': 10000, 'max': 200000, 'default': 84000},
    {'name': 'current_address_months_count', 'label': 'Current Address Months Count', 'type': 'number', 'min': 0, 'max': 600, 'default': 50},
    {'name': 'customer_age', 'label': 'Customer Age', 'type': 'number', 'min': 10, 'max': 100, 'default': 30},
    {'name': 'days_since_request', 'label': 'Days Since Request', 'type': 'number', 'min': 0, 'max': 365, 'default': 30}, # cook this variable it stinks
    {'name': 'payment_type', 'label': 'Payment Type', 'type': 'categorical', 'options': ['AA', 'AB', 'AC', 'AD', 'AE', 'Missing'], 'default': 'AB'},
    {'name': 'zip_count_4w', 'label': 'Zip Count 4w', 'type': 'number', 'min': 1, 'max': 6000, 'default': 1200},
    {'name': 'velocity_6h', 'label': 'Velocity 6h', 'type': 'number', 'min': 0, 'max': 17000, 'default': 5300},
    {'name': 'velocity_24h', 'label': 'Velocity 24h', 'type': 'number', 'min': 1300, 'max': 10000, 'default': 4700},
    {'name': 'velocity_4w', 'label': 'Velocity 4w', 'type': 'number', 'min': 2800, 'max': 7000, 'default': 4900},
    {'name': 'bank_branch_count_8w', 'label': 'Bank Branch Count 8w', 'type': 'number', 'min': 0, 'max': 2400, 'default': 9},
    {'name': 'date_of_birth_distinct_emails_4w', 'label': 'DOB Distinct Emails 4w', 'type': 'number', 'min': 0, 'max': 40, 'default': 9},
    {'name': 'employment_status', 'label': 'Employment Status', 'type': 'categorical', 'options': ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'Missing'], 'default': 'CA'},
    {'name': 'credit_risk_score', 'label': 'Credit Risk Score', 'type': 'number', 'min': -170, 'max': 390, 'default': 120},
    {'name': 'email_is_free', 'label': 'Email Is Free', 'type': 'boolean', 'default': True},
    {'name': 'housing_status', 'label': 'Housing Status', 'type': 'categorical', 'options': ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'Missing'], 'default': 'BC'},
    {'name': 'phone_home_valid', 'label': 'Phone Home Valid', 'type': 'boolean', 'default': True},
    {'name': 'phone_mobile_valid', 'label': 'Phone Mobile Valid', 'type': 'boolean', 'default': True},
    {'name': 'has_other_cards', 'label': 'Has Other Cards', 'type': 'boolean', 'default': False},
    {'name': 'proposed_credit_limit', 'label': 'Proposed Credit Limit', 'type': 'number', 'min': 200, 'max': 2000, 'default': 200},
    {'name': 'foreign_request', 'label': 'Foreign Request', 'type': 'boolean', 'default': False},
    {'name': 'source', 'label': 'Source', 'type': 'categorical', 'options': ['INTERNET', 'TELEAPP', 'Missing'], 'default': 'INTERNET'},
    {'name': 'session_length_in_minutes', 'label': 'Session Length (Minutes)', 'type': 'number', 'min': 0, 'max': 90, 'default': 5},
    {'name': 'device_os', 'label': 'Device OS', 'type': 'categorical', 'options': ['linux', 'macintosh', 'Missing', 'other', 'windows', 'x11'], 'default': 'other'},
    {'name': 'keep_alive_session', 'label': 'Keep Alive Session', 'type': 'boolean', 'default': True},
    {'name': 'device_distinct_emails_8w', 'label': 'Device Distinct Emails 8w', 'type': 'number', 'min': 0, 'max': 2, 'default': 1},
    {'name': 'month', 'label': 'Month', 'type': 'categorical', 'options': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'Missing'], 'default': 'January'},
    {'name': 'email_domain', 'label': 'Email Domain', 'type': 'categorical', 'options': ['aol.com', 'agency.io', 'zoho.com', 'protonmail.com', 'business.org', 'yandex.com', 'consulting.co', 'icloud.com', 'work.net', 'finance.pro', 'lawfirm.legal', 'gmx.com', 'company.com', 'startup.biz', 'tech.info', 'Missing', 'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'other'], 'default': 'gmail.com'},
    {'name': 'prev_address_count_bucket', 'label': 'Previous Address Count Bucket', 'type': 'categorical', 'options': ['0-12 Months', '1-2 Years', '3-5 Years', '5-10 Years', '10+ Years', 'Unknown'], 'default': 'Unknown'},
    {'name': 'estimated_age', 'label': 'Estimated Age', 'type': 'number', 'min': 10, 'max': 100, 'default': 35},
    {'name': 'bank_months_count_bucket', 'label': 'Bank Months Count Bucket', 'type': 'categorical', 'options': ['1 Month', '1-3 Months', '3-12 Months', '12-24 Months', '24+ Months', 'Unknown'], 'default': 'Unknown'},
    {'name': 'intended_balcon_amount_bucket', 'label': 'Intended Balance Amount Bucket', 'type': 'categorical', 'options': ['<$20', '$20-$40', '$40-$80', '$80-$100', '$100+', 'Unknown'], 'default': 'Unknown'},
]

# Title
st.title("🎯 Logistic Regression Predictor")
st.markdown("---")

# Sidebar for feature selection
with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("Select Features to Display")
    
    # Quick buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✓ Show All"):
            for feature in ALL_FEATURES:
                st.session_state[f"show_{feature['name']}"] = True
    with col2:
        if st.button("✗ Hide All"):
            for feature in ALL_FEATURES:
                st.session_state[f"show_{feature['name']}"] = False
    
    st.markdown("---")
    
    # Feature checkboxes
    for feature in ALL_FEATURES:
        key = f"show_{feature['name']}"
        if key not in st.session_state:
            st.session_state[key] = True
        st.checkbox(feature['label'], key=key)

# Main content area
st.header("📊 Input Features")

# Collect input values
input_values = {}

# Create columns for inputs
visible_features = [f for f in ALL_FEATURES if st.session_state.get(f"show_{f['name']}", True)]

if len(visible_features) == 0:
    st.warning("⚠️ No features selected. Please enable at least one feature in the sidebar.")
else:
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    for idx, feature in enumerate(visible_features):
        # Alternate between columns
        current_col = col1 if idx % 2 == 0 else col2
    
        with current_col:
            if feature['type'] == 'boolean':
                value = st.checkbox(
                    feature['label'],
                    value=feature['default'],
                    key=f"input_{feature['name']}"
                )
                input_values[feature['name']] = 1 if value else 0
            elif feature['type'] == 'categorical':
                value = st.selectbox(
                    feature['label'],
                    options=feature['options'],
                    index=feature['options'].index(feature['default']),
                    key=f"input_{feature['name']}"
                )
                input_values[feature['name']] = value
            else:  # number type
                value = st.number_input(
                    feature['label'],
                    min_value=feature['min'],
                    max_value=feature['max'],
                    value=feature['default'],
                    key=f"input_{feature['name']}"
                )
                input_values[feature['name']] = float(value)

    # Add all features with default values if they're not visible
    # Add all features with default values if they're not visible
    for feature in ALL_FEATURES:
        if feature['name'] not in input_values:
            if feature['type'] == 'boolean':
                input_values[feature['name']] = 1 if feature['default'] else 0
            elif feature['type'] == 'categorical':
                input_values[feature['name']] = feature['default']
            else:
                input_values[feature['name']] = float(feature['default'])

    st.markdown("---")

    # Make prediction
    st.header("🎲 Prediction Result")
    
    #probability = model.predict_proba(input_values)
    # Prepare input for model
    # Prepare input for model
    model_input = prepare_model_input(input_values, ALL_FEATURES)
    # Add constant (intercept) - statsmodels needs this
    model_input = sm.add_constant(model_input, has_constant='add')
    # Ensure the column order matches what the model expects
    model_input = model_input[loaded_model.model.exog_names]
    # Ensure all values are float
    model_input = model_input.astype(float)
    # Make prediction
    prediction_proba = loaded_model.predict(model_input)[0]
    probability = float(prediction_proba)

    predicted_class = "Positive" if probability >= 0.5 else "Negative"
    
    # Display prediction in columns
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.metric(
            label="Probability of Positive Class",
            value=f"{probability * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Predicted Class",
            value=predicted_class
        )
    
    with col3:
        # Color indicator
        if probability >= 0.5:
            st.success("✓")
        else:
            st.error("✗")
    
    # Progress bar
    st.progress(probability)

    # Do a cost benefit analysis based on direct cost of accepting fraud and the CLV hit of turning away real business
    st.header("⚖️ Cost Benefit Analysis")

    # Display the input boxes in columns
    fraudCost, cusLoss, acquiCost = st.columns([2, 2, 1])

    with fraudCost:
        value = st.number_input(
                    'Cost of Accepting Fraud',
                    min_value=100,
                    max_value=20000,
                    value=1200,
                    key=f"input_fraudCost"
                )
        input_values['fraudCost'] = float(value)

    with cusLoss:
        value = st.number_input(
                    'Customer Lifetime Value',
                    min_value=0,
                    max_value=20000,
                    value=1000,
                    key=f"input_cusLoss"
                )
        input_values['cusLTV'] = float(value)

    with acquiCost:
        value = st.number_input(
                    'Cost per Acquisition',
                    min_value=0,
                    max_value=400,
                    value=165,
                    key=f"input_acquiCost"
                )
        input_values['CPA'] = float(value)
    
    # Now display the expected values for each scenario based on the probability that is input
    ev_accept = input_values['cusLTV'] * (1-probability) - input_values['CPA'] - probability * input_values['fraudCost']
    ev_reject = -1 * input_values['cusLTV'] * (1-probability) - input_values['CPA']

    # Determine which of these values is higher
    prefer_accept = ev_accept > ev_reject

    ev_fraud, ev_cus = st.columns([2, 2])

    with ev_fraud:
        if prefer_accept:
            st.markdown(f"""
                <div style='padding: 20px; background-color: #d4edda; border-radius: 10px; border: 2px solid #28a745;'>
                    <h4 style='color: #155724; margin: 0;'>Expected Value of Accepting this Customer</h4>
                    <h2 style='color: #155724; margin: 10px 0 0 0;'>${ev_accept:.2f}</h2>
                </div>
        """, unsafe_allow_html=True)
        else:
            #st.metric("Expected Value of Accepting this Customer", f"${ev_accept:.2f}")
            st.markdown(f"""
                <div style='padding: 20px; background-color: #eec2c2; border-radius: 10px; border: 2px solid #a72828;'>
                    <h4 style='color: #6c1b1b; margin: 0;'>Expected Value of Accepting this Customer</h4>
                    <h2 style='color: #6c1b1b; margin: 10px 0 0 0;'>${ev_accept:.2f}</h2>
                </div>
        """, unsafe_allow_html=True)

    with ev_cus:
        if not prefer_accept:
            st.markdown(f"""
                <div style='padding: 20px; background-color: #d4edda; border-radius: 10px; border: 2px solid #28a745;'>
                    <h4 style='color: #155724; margin: 0;'>Expected Value of Rejecting this Customer</h4>
                    <h2 style='color: #155724; margin: 10px 0 0 0;'>${ev_reject:.2f}</h2>
                </div>
        """, unsafe_allow_html=True)
        else:
            #st.metric("Expected Value of Accepting this Customer", f"${ev_accept:.2f}")
            st.markdown(f"""
                <div style='padding: 20px; background-color: #eec2c2; border-radius: 10px; border: 2px solid #a72828;'>
                    <h4 style='color: #6c1b1b; margin: 0;'>Expected Value of Rejecting this Customer</h4>
                    <h2 style='color: #6c1b1b; margin: 10px 0 0 0;'>${ev_reject:.2f}</h2>
                </div>
        """, unsafe_allow_html=True)

    # Additional info in expander
    with st.expander("📋 View Input Summary"):
        st.json(input_values)

# Instructions
st.markdown("---")
st.subheader("📖 How to Use")
st.markdown("""
1. **Adjust Input Values**: Use the input fields to enter values for each feature
2. **Toggle Features**: Use the sidebar to show/hide specific features and reduce clutter
3. **View Prediction**: The prediction updates automatically as you change values
4. **Replace Mock Model**: Update the `load_model()` function with your actual trained model

**To use your own model:**
```python
@st.cache_resource
def load_model():
    with open('your_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
```

Then use `model.predict_proba()` with your feature values formatted as needed by your model.
""")