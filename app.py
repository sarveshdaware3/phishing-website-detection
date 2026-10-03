import streamlit as st
import joblib
import pandas as pd
import re
from urllib.parse import urlparse

# Load trained model
model = joblib.load("best_phishing_model.pkl")

st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🔐",
    layout="centered"
)

st.title("🔐 Phishing Website Detection")
st.write("Enter a website URL to check whether it is legitimate or phishing.")

url = st.text_input(
    "Website URL",
    placeholder="https://example.com"
)


def extract_url_features(url):
    parsed = urlparse(url)

    hostname = parsed.hostname or ""

    features = {}

    features["length_url"] = len(url)
    features["length_hostname"] = len(hostname)

    features["ip"] = int(
        bool(
            re.match(
                r"^(?:\d{1,3}\.){3}\d{1,3}$",
                hostname
            )
        )
    )

    features["nb_dots"] = url.count(".")
    features["nb_hyphens"] = url.count("-")
    features["nb_at"] = url.count("@")
    features["nb_qm"] = url.count("?")
    features["nb_and"] = url.count("&")
    features["nb_or"] = url.count("|")
    features["nb_eq"] = url.count("=")
    features["nb_underscore"] = url.count("_")
    features["nb_tilde"] = url.count("~")
    features["nb_percent"] = url.count("%")
    features["nb_slash"] = url.count("/")
    features["nb_star"] = url.count("*")
    features["nb_colon"] = url.count(":")
    features["nb_comma"] = url.count(",")
    features["nb_semicolumn"] = url.count(";")
    features["nb_dollar"] = url.count("$")
    features["nb_space"] = url.count(" ")
    features["nb_www"] = url.lower().count("www")
    features["nb_com"] = url.lower().count(".com")
    features["nb_dslash"] = url.count("//")

    return pd.DataFrame([features])


if st.button("Check URL"):

    if not url.strip():

        st.warning("Please enter a website URL.")

    else:

        try:

            if not url.startswith(("http://", "https://")):
                url = "http://" + url

            input_features = extract_url_features(url)

            # Get features expected by the trained model
            expected_features = model.feature_names_in_

            # Create missing features with zero values
            for feature in expected_features:

                if feature not in input_features.columns:
                    input_features[feature] = 0

            # Keep only training features and correct order
            input_features = input_features[
                expected_features
            ]

            prediction = model.predict(
                input_features
            )[0]

            if prediction == 1:

                st.error(
                    "⚠️ PHISHING WEBSITE DETECTED"
                )

                st.write(
                    "This URL is classified as potentially phishing."
                )

            else:

                st.success(
                    "✅ LEGITIMATE WEBSITE"
                )

                st.write(
                    "This URL is classified as legitimate."
                )

        except Exception as e:

            st.error(
                "Unable to check this URL."
            )

            st.write(
                "Error:",
                str(e)
            )