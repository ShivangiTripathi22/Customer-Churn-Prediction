# import streamlit as st
# import pandas as pd
# import joblib

# # Load model
# model = joblib.load("customer_churn_model.pkl")

# st.title("🏦 Customer Churn Prediction")
# st.write("Predict whether a customer is likely to churn.")

# # User inputs
# credit_score = st.number_input(
#     "Credit Score",
#     min_value=300,
#     max_value=850,
#     value=650
# )

# geography = st.selectbox(
#     "Geography",
#     ["France", "Germany", "Spain"]
# )

# gender = st.selectbox(
#     "Gender",
#     ["Male", "Female"]
# )

# age = st.number_input(
#     "Age",
#     min_value=18,
#     max_value=100,
#     value=40
# )

# tenure = st.number_input(
#     "Tenure",
#     min_value=0,
#     max_value=10,
#     value=5
# )

# balance = st.number_input(
#     "Balance",
#     min_value=0.0,
#     value=50000.0
# )

# num_products = st.number_input(
#     "Number of Products",
#     min_value=1,
#     max_value=4,
#     value=1
# )

# has_card = st.selectbox(
#     "Has Credit Card",
#     ["Yes", "No"]
# )

# active_member = st.selectbox(
#     "Is Active Member",
#     ["Yes", "No"]
# )

# salary = st.number_input(
#     "Estimated Salary",
#     min_value=0.0,
#     value=50000.0
# )

# # Convert Yes/No to 1/0
# has_card_value = 1 if has_card == "Yes" else 0
# active_member_value = 1 if active_member == "Yes" else 0

# # Create input dataframe
# input_data = pd.DataFrame({
#     "CreditScore": [credit_score],
#     "Geography": [geography],
#     "Gender": [gender],
#     "Age": [age],
#     "Tenure": [tenure],
#     "Balance": [balance],
#     "NumOfProducts": [num_products],
#     "HasCrCard": [has_card_value],
#     "IsActiveMember": [active_member_value],
#     "EstimatedSalary": [salary]
# })

# # Prediction
# if st.button("Predict Churn"):

#     prediction = model.predict(input_data)[0]
#     probability = model.predict_proba(input_data)[0][1]

#     if prediction == 1:
#         st.error("⚠️ Customer is likely to churn")
#     else:
#         st.success("✅ Customer is likely to stay")

#     st.metric(
#         "Churn Probability",
#         f"{probability:.2%}"
#     )






import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM DARK THEME
# =========================================================

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------------------------------------------
       HEADINGS
    --------------------------------------------- */

    h1 {
        color: #f8fafc !important;
        font-weight: 750 !important;
    }

    h2, h3 {
        color: #f8fafc !important;
    }

    /* Caption text */
    .stCaption {
        color: #94a3b8 !important;
    }

    /* ---------------------------------------------
       INPUT LABELS
    --------------------------------------------- */

    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    /* ---------------------------------------------
       TEXT INPUT / NUMBER INPUT
    --------------------------------------------- */

    div[data-baseweb="input"] {
        background-color: #172033 !important;
        border-radius: 8px !important;
    }

    div[data-baseweb="input"] input {
        color: #f8fafc !important;
    }

    /* ---------------------------------------------
       SELECTBOX
    --------------------------------------------- */

    div[data-baseweb="select"] > div {
        background-color: #172033 !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }

    /* ---------------------------------------------
       BUTTON
    --------------------------------------------- */

    div.stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 10px;
        border: none;
        background-color: #2563eb;
        color: white;
        font-size: 16px;
        font-weight: 700;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
    }

    /* ---------------------------------------------
       METRIC CARDS
    --------------------------------------------- */

    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* ---------------------------------------------
       SUCCESS / ERROR BOX
    --------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ---------------------------------------------
       PROGRESS BAR
    --------------------------------------------- */

    div[data-testid="stProgress"] {
        margin-top: 10px;
        margin-bottom: 10px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")


try:
    model = load_model()

except Exception:
    st.error("❌ Unable to load the customer churn model.")

    st.warning(
        "Make sure that `customer_churn_model.pkl` is present "
        "in the same folder as `app.py`."
    )

    st.stop()


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🏦 Customer Churn Prediction")

st.caption(
    "Predict customer retention risk using machine learning"
)

st.divider()


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.subheader("Customer Profile")

st.caption(
    "Enter the customer's demographic and account information "
    "to estimate the probability of churn."
)


# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650,
        step=1
    )


with col2:

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )


with col3:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# =========================================================
# ROW 2
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )


with col2:

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )


with col3:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        max_value=300000.0,
        value=50000.0,
        step=1000.0
    )


# =========================================================
# ROW 3
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1,
        step=1
    )


with col2:

    has_credit_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )


with col3:

    active_member = st.selectbox(
        "Active Member",
        ["Yes", "No"]
    )


# =========================================================
# ROW 4
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        max_value=250000.0,
        value=50000.0,
        step=1000.0
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])


with button_col2:

    predict_button = st.button(
        "🔮  PREDICT CHURN",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Convert categorical Yes/No values
    # -----------------------------------------------------

    has_credit_card_value = (
        1 if has_credit_card == "Yes" else 0
    )

    active_member_value = (
        1 if active_member == "Yes" else 0
    )


    # -----------------------------------------------------
    # Create input dataframe
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "CreditScore": [credit_score],

        "Geography": [geography],

        "Gender": [gender],

        "Age": [age],

        "Tenure": [tenure],

        "Balance": [balance],

        "NumOfProducts": [num_products],

        "HasCrCard": [has_credit_card_value],

        "IsActiveMember": [active_member_value],

        "EstimatedSalary": [estimated_salary]

    })


    # -----------------------------------------------------
    # Make prediction
    # -----------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        probability_percent = probability * 100


        # =================================================
        # RESULT SECTION
        # =================================================

        st.divider()

        st.subheader("Prediction Result")

        st.caption(
            "Machine learning assessment of this customer's "
            "churn risk."
        )


        # =================================================
        # RISK RESULT
        # =================================================

        if prediction == 1:

            st.error(
                f"⚠️ HIGH CHURN RISK — "
                f"{probability_percent:.1f}%"
            )

            st.progress(float(probability))

        else:

            st.success(
                f"✓ LOW CHURN RISK — "
                f"{probability_percent:.1f}%"
            )

            st.progress(float(probability))


        st.caption("Estimated Churn Probability")


        # =================================================
        # SUMMARY METRICS
        # =================================================

        st.write("")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                label="Prediction",
                value=(
                    "CHURN"
                    if prediction == 1
                    else "STAY"
                )
            )


        with col2:

            st.metric(
                label="Churn Probability",
                value=f"{probability_percent:.1f}%"
            )


        with col3:

            st.metric(
                label="Customer Age",
                value=f"{age} years"
            )


        # =================================================
        # BUSINESS RECOMMENDATION
        # =================================================

        st.write("")

        if prediction == 1:

            st.warning(
                "💡 **Recommended Action:** "
                "This customer shows a higher likelihood of leaving. "
                "Consider targeted retention strategies such as "
                "personalized offers, proactive customer support, "
                "or relationship-management outreach."
            )

        else:

            st.info(
                "💡 **Recommended Action:** "
                "This customer currently shows a lower likelihood "
                "of churn. Continue maintaining engagement and "
                "consistent customer service."
            )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.warning(
            "Please verify that the model was trained using the "
            "same input features used by this application."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.caption(
    "Customer Churn Prediction • Machine Learning Dashboard"
)