import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from joblib import load

# Download NLTK resources if not already downloaded
nltk.download('names')

# Feature extraction function
def extract_gender_features(name):
    name = name.lower()
    features = {
        "suffix": name[-1:],
        "suffix2": name[-2:] if len(name) > 1 else name[0],
        "suffix3": name[-3:] if len(name) > 2 else name[0],
        "suffix4": name[-4:] if len(name) > 3 else name[0],
        "suffix5": name[-5:] if len(name) > 4 else name[0],
        "suffix6": name[-6:] if len(name) > 5 else name[0],
        "prefix": name[:1],
        "prefix2": name[:2] if len(name) > 1 else name[0],
        "prefix3": name[:3] if len(name) > 2 else name[0],
        "prefix4": name[:4] if len(name) > 3 else name[0],
        "prefix5": name[:5] if len(name) > 4 else name[0]
    }
    return features

# Load trained model
bayes = load('gender_prediction.joblib')

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        font-size: 3rem;
        color: #4B77BE;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .sub-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 400;
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #4B77BE;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1.1rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3A5D99;
        color: #fff;
    }
    .result-box {
        background-color: #eaf3fb;
        border-left: 6px solid #4B77BE;
        padding: 1em 1.2em;
        border-radius: 8px;
        font-size: 1.4rem;
        margin-top: 1em;
        font-weight: 600;
        color: #222;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🚻 Gender Prediction App</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter a name below to predict its gender with high accuracy.</p>', unsafe_allow_html=True)

    # Using form for cleaner input/submit flow
    with st.form(key='name_form'):
        input_name = st.text_input('Your Name', max_chars=30, placeholder="Type a name here...")
        submit_button = st.form_submit_button(label='Predict Gender')

    if submit_button:
        if input_name.strip():
            features = extract_gender_features(input_name)
            predicted_gender = bayes.classify(features)

            st.markdown(f'<div class="result-box">Prediction: The name <b>"{input_name}"</b> is most likely <span style="color:#4B77BE;">{predicted_gender}</span>.</div>', unsafe_allow_html=True)
        else:
            st.warning('⚠️ Please enter a valid name.')

if __name__ == "__main__":
    main()
