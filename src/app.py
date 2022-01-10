"""Basic web app to explore the performance of the ML app."""
import joblib
import pandas as pd
import streamlit as st

import config

st.set_page_config(layout="wide")


@st.cache
def load_categories():
    """Load the list of categories.

    Returns:
        a DataFrame.
    """
    return pd.read_csv("{0}categories.csv".format(config.DOCS))


@st.cache(allow_output_mutation=True)
def load_model():
    """Load Machine and returns Learning model.

    Returns:
        an instance of sklearn classifier.
    """
    return joblib.load("{0}best_model.z".format(config.MODELS))


@st.cache
def load_features():
    """Load the required features for predictions.

    Returns:
        a DataFrame
    """
    return pd.read_csv("{0}best_features.csv".format(config.DOCS))


@st.cache
def load_samples():
    """Load the samples exemples.

    Returns:
        a string content as markdown
    """
    with open("{0}samples.md".format(config.DOCS)) as samples:
        return samples.read()


model = load_model()
categories = load_categories()

# 1 : get data from User
features = pd.read_csv("{0}best_features.csv".format(config.DOCS))["feature"]

# App title
st.title("Nutriscore App predictor")

with st.expander("Try with product data from OpenFoodsFacts:"):
    st.markdown(load_samples())

# Organization of the app
main_block = st.columns([1])

energy = st.text_input(label="Energy:", value=0)

saturated_fat = st.text_input(label="Saturated fat:", value=0)

sugars = st.text_input(label="Sugars:", value=0)

salt = st.text_input(label="Salt:", value=0)

fibers = st.text_input(label="Fibers:", value=0)

category = st.selectbox(label="Category:", options=categories)

product = {
    "energy": round(float(energy)),
    "saturated_fat": round(float(saturated_fat)),
    "sugars": round(float(sugars)),
    "salt": round(float(salt)),
    "fibers": round(float(fibers)),
    "group1_Beverages": 0,
    "group1_Cereals and potatoes": 0,
    "group1_Composite foods": 0,
    "group1_Fat and sauces": 0,
    "group1_Fruits and vegetables": 0,
    "group1_Milk and dairy products": 0,
    "group1_Sugary snacks": 0,
    "group1_unknown": 0,
}

# If category is detected then assign the property value to 1.
formatted_category = "group1_{0}".format(category)

if formatted_category in product.keys():
    product[formatted_category] = 1

sample = list(product.values())

# Predict the nutrigrade !
nutrigrade = model.predict([sample])

st.write(
    "The predicted nutrigrade for this product is : **{0}**".format(
        nutrigrade[0].upper(),
    ),
)
