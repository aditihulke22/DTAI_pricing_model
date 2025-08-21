import pickle

# Define the path to your pickle file
model_pkl_file = "model.pkl"

# Open the pickle file in binary read mode ('rb')
with open(model_pkl_file, 'rb') as file:
    # Load the model using pickle.load()
    model = pickle.load(file)

# Now, 'loaded_model' contains your deserialized model object,
# ready for use (e.g., for making predictions).

UPC_v = 76672750557
STORE_v = 144.0
WEEK_v = 353
SIZE_v = 15
PRICE_v = 0.5


features = [[UPC_v, STORE_v, WEEK_v, SIZE_v, PRICE_v]]
sales = model.predict(features)[0]

print("The predicted sales for product id " + str(UPC_v) + " in week " + str(WEEK_v) + " and store number " + str(STORE_v) + " is " + str(round(sales)) + " units")
print("The corresponding SKU size is " + str(SIZE_v) + " Oz,at per Oz price of $" + str(PRICE_v))
