import streamlit as st
import pickle
import pandas as pd

# -- Load the trained model once --
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

# -- UI components --
st.title("Sales Prediction")

# Example user input fields; edit as per your features
product_id = st.number_input("Product ID")
sku_size = st.number_input("SKU Size", min_value=1, max_value=1000)
price = st.number_input("Price", min_value=0.01)
store = st.text_input("Store ID")
week = st.number_input("Week")
#st.write(type(product_id))


if st.button("Predict Sales"):
    # Process input into required DataFrame
    features = pd.DataFrame([[float(product_id), float(store), float(week), float(sku_size), float(price)]],
                            columns=["UPC", "STORE", "WEEK", "SIZE_FLOAT", "AVG_PRICE_PER_OZ"])
    # Encode categorical if needed (e.g., store), preprocess as per training!
    # features['store'] = encoder.transform(features[['store']])
    # -- Predict --
    prediction = model.predict(features)[0]

    if prediction < 0:
        prediction = 0

    #st.write(f"Predicted Sales: {prediction[0]:.2f}")
    st.write("The predicted sales for product id " + str((product_id)) + " in week " + str(week) + " and store number " + str(store) + " is " + str(round(prediction)) + " units")
    if prediction == 0:
            st.write("The likehood of sales is very less.")
    st.write("The corresponding SKU size is " + str(sku_size) + " Oz,at per Oz price of $" + str(price))
