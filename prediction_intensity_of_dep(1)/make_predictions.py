from sklearn.externals import joblib

# Load the model we trained previously
model = joblib.load('trained_Intensity of deprivation Regional_classifier_model.pkl')

# For the house we want to value, we need to provide the features in the exact same
# arrangement as our training data set.
yemen_to_value = [
   0.035,
   8.5
]

# scikit-learn assumes you want to predict the values for lots of houses at once, so it expects an array.
# We just want to look at a single house, so it will be the only item in our array.
regions_to_value = [
    yemen_to_value
]

# Run the model and make a prediction for each house in the homes_to_value array
predicted_region_values = model.predict(regions_to_value)

# Since we are only predicting the price of one house, just look at the first prediction returned
predicted_value = predicted_region_values[0]

print("This region has an estimated value of deprivation ${:,.2f}".format(predicted_value))

