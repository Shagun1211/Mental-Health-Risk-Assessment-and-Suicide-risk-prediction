import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
@st.cache_resource
def load_models():

    # ✅ NEW: Lite Reddit model
    reddit_model = pickle.load(open("reddit_model_lite.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

    # Survey model
    survey_model = pickle.load(open("survey_model.pkl", "rb"))
    le_target = pickle.load(open("survey_label_encoder.pkl", "rb"))
    feature_cols = pickle.load(open("feature_cols.pkl", "rb"))

    return reddit_model, tfidf, survey_model, le_target, feature_cols


reddit_model, tfidf, survey_model, le_target, feature_cols = load_models()

# Page config
st.title("🧠 Mental Health Risk Prediction")
st.write("Enter your details below to get a risk assessment.")

# ---------------------------
# Reddit Prediction Function
# ---------------------------
def predict_reddit(text):
    vec = tfidf.transform([text])
    prob = reddit_model.predict_proba(vec)[0][1]
    return prob

# ---------------------------
# UI
# ---------------------------
st.header("1. How are you feeling?")
user_text = st.text_area(
    "Write something about how you feel today:",
    placeholder="e.g. I feel stressed and overwhelmed..."
)

st.header("2. Lifestyle Survey")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 22)
    sleep_hours = st.slider("Sleep hours per night", 1, 12, 7)
    stress = st.slider("Stress level (1-5)", 1, 5, 3)
    loneliness = st.slider("Loneliness level (1-5)", 1, 5, 3)
    screen_time = st.slider("Screen time (hours/day)", 1, 12, 4)

with col2:
    family_support = st.slider("Family support (1-5)", 1, 5, 3)
    physical_activity = st.selectbox("Physical activity", ['High', 'Medium', 'Low'])
    alcohol = st.selectbox("Alcohol use", ['No', 'Yes'])
    smoking = st.selectbox("Smoking", ['No', 'Yes'])
    employment = st.selectbox("Employment", ['Employed', 'Unemployed', 'Student'])

gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
education = st.selectbox("Education", ['School', 'College', 'Graduate', 'Postgraduate'])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Get Risk Assessment"):

    if not user_text:
        st.warning("Please enter some text about how you feel!")
    else:
        with st.spinner("Analyzing..."):

            # ✅ Reddit prediction (NEW)
            reddit_prob = float(predict_reddit(user_text))

            # ---------------------------
            # Survey Feature Engineering
            # ---------------------------
            activity_map = {'High': 2, 'Medium': 1, 'Low': 0}
            activity_num = activity_map[physical_activity]

            gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
            edu_map = {'School': 2, 'College': 0, 'Graduate': 1, 'Postgraduate': 3}
            emp_map = {'Employed': 0, 'Unemployed': 2, 'Student': 1}
            alc_map = {'No': 0, 'Yes': 1}
            smk_map = {'No': 0, 'Yes': 1}

            # Engineered features
            risk_score = loneliness + stress - family_support
            sleep_stress_ratio = stress / (sleep_hours + 0.1)
            lifestyle_index = sleep_hours - screen_time + activity_num
            substance_use = alc_map[alcohol] + smk_map[smoking]
            stress_x_loneliness = stress * loneliness
            support_activity = family_support * activity_num
            sleep_deficit = max(0, 7 - sleep_hours)
            high_risk_flag = int(stress >= 4 and loneliness >= 4 and sleep_hours <= 5)

            survey_input = {
                'age': age,
                'gender': gender_map[gender],
                'education': edu_map[education],
                'employment': emp_map[employment],
                'sleep_hours': sleep_hours,
                'screen_time': screen_time,
                'physical_activity': activity_num,
                'alcohol': alc_map[alcohol],
                'smoking': smk_map[smoking],
                'family_support': family_support,
                'loneliness': loneliness,
                'stress': stress,
                'suicide_risk': 0,
                'activity_num': activity_num,
                'risk_score': risk_score,
                'sleep_stress_ratio': sleep_stress_ratio,
                'lifestyle_index': lifestyle_index,
                'substance_use': substance_use,
                'stress_x_loneliness': stress_x_loneliness,
                'support_activity': support_activity,
                'sleep_deficit': sleep_deficit,
                'high_risk_flag': high_risk_flag
            }

            df_input = pd.DataFrame([survey_input])[feature_cols]

            survey_probs = survey_model.predict_proba(df_input)[0]
            survey_pred = survey_model.predict(df_input)[0]
            survey_label = le_target.inverse_transform([survey_pred])[0]

            # Get class probabilities
            class_list = list(le_target.classes_)
            survey_high = float(survey_probs[class_list.index('High')])
            survey_low = float(survey_probs[class_list.index('Low')])

            # ---------------------------
            # Fusion
            # ---------------------------
            fused_high = (0.4 * reddit_prob) + (0.6 * survey_high)
            fused_low = (0.4 * (1 - reddit_prob)) + (0.6 * survey_low)
            fused_med = max(0, 1 - fused_high - fused_low)

            # Normalize
            total = fused_high + fused_med + fused_low
            fused_high /= total
            fused_med /= total
            fused_low /= total

            # ---------------------------
            # Display Results
            # ---------------------------
            st.header("Results")

            if fused_high >= 0.6:
                st.error("🔴 High Risk")
            elif fused_med >= 0.4:
                st.warning("🟡 Medium Risk")
            else:
                st.success("🟢 Low Risk")

            col1, col2, col3 = st.columns(3)
            col1.metric("High Risk", f"{fused_high*100:.1f}%")
            col2.metric("Medium Risk", f"{fused_med*100:.1f}%")
            col3.metric("Low Risk", f"{fused_low*100:.1f}%")

            st.subheader("Why this result?")

            reasons = []

            # Survey-based explanations
            if stress >= 4:
                reasons.append("High stress level detected")

            if loneliness >= 4:
                reasons.append("High loneliness reported")

            if sleep_hours <= 5:
                reasons.append("Low sleep duration")

            if family_support <= 2:
                reasons.append("Low family support")

            if screen_time >= 8:
                reasons.append("High screen time")

            # Reddit text explanation
            if reddit_prob > 0.7:
                reasons.append("Concerning emotional tone detected in text")

            elif reddit_prob < 0.3:
                reasons.append("Neutral or positive emotional tone detected")

            # Default
            if len(reasons) == 0:
                reasons.append("Overall balanced lifestyle indicators")

            for r in reasons:
                st.write("•", r)

            if fused_high >= 0.6:
                st.info("💙 If you are struggling, please reach out to iCall: 9152987821")