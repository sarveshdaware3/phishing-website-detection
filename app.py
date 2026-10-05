
import streamlit as st
import joblib
import pandas as pd
from urllib.parse import urlparse


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔐",
    layout="centered"
)


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

MODEL_PATH = "best_phishing_model.pkl"
FEATURE_PATH = "model_features.pkl"


try:
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURE_PATH)

except Exception as e:
    st.error(f"Model files could not be loaded: {e}")
    st.stop()


# ---------------------------------------------------------
# URL Feature Extraction
# ---------------------------------------------------------

def extract_url_features(url):

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)

    hostname = parsed.hostname or ""

    features = {
        "length_url": len(url),
        "length_hostname": len(hostname),

        "ip": int(
            hostname.replace(".", "").isdigit()
        ),

        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" "),

        "nb_www": url.lower().count("www"),
        "nb_com": url.lower().count(".com"),
        "nb_dslash": url.count("//"),

        "engineered_url_length": len(url),

        "special_char_count": sum(
            not c.isalnum()
            for c in url
        ),

        "digit_count_url": sum(
            c.isdigit()
            for c in url
        ),

        "dot_count_url": url.count("."),

        "slash_count_url": url.count("/"),

        "has_at_symbol": int("@" in url),

        "has_hyphen": int("-" in url)
    }

    return features


# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------

st.title("🔐 Phishing Website Detection")

st.write(
    "Enter a website URL to check whether it is legitimate or phishing."
)

url = st.text_input(
    "Website URL",
    placeholder="https://example.com"
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Check Website"):

    if not url.strip():

        st.warning("Please enter a website URL.")

    else:

        try:

            # Extract URL features
            features = extract_url_features(url)

            # Convert to DataFrame
            input_df = pd.DataFrame([features])

            # Ensure exact model feature order
            for feature in model_features:

                if feature not in input_df.columns:
                    input_df[feature] = 0

            input_df = input_df[model_features]

            # Model prediction
            prediction = model.predict(input_df)[0]

            # -------------------------------------------------
            # Probability
            # -------------------------------------------------

            phishing_probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_df)[0]

                # Find probability for class 1
                if hasattr(model, "classes_"):

                    classes = list(model.classes_)

                    if 1 in classes:
                        phishing_probability = probabilities[
                            classes.index(1)
                        ]

                    elif "phishing" in classes:
                        phishing_probability = probabilities[
                            classes.index("phishing")
                        ]

                else:

                    phishing_probability = probabilities[-1]

            # -------------------------------------------------
            # Result
            # -------------------------------------------------

            st.markdown("---")

            if int(prediction) == 1:

                st.error("⚠️ PHISHING WEBSITE")

                st.write(
                    "This URL is predicted to be a phishing website."
                )

            else:

                st.success("✅ LEGITIMATE WEBSITE")

                st.write(
                    "This URL is predicted to be a legitimate website."
                )

            # -------------------------------------------------
            # Probability Display
            # -------------------------------------------------

            if phishing_probability is not None:

                phishing_percent = phishing_probability * 100
                legitimate_percent = 100 - phishing_percent

                st.subheader("Prediction Probability")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Phishing",
                        f"{phishing_percent:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Legitimate",
                        f"{legitimate_percent:.2f}%"
                    )

                st.progress(
                    min(max(float(phishing_probability), 0.0), 1.0)
                )

            else:

                st.info(
                    "Probability is not available for this saved model."
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Phishing Website Detection using Machine Learning"
)
