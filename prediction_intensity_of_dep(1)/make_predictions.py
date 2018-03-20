from sklearn.externals import joblib

# Load the model we trained previously
model = joblib.load('trained_Intensity of deprivation Regional_classifier_model.pkl')

# For the Afghanistan_Badakhshan (sample from Afghanistan) we want to value, we need to provide the features in the exact same
# arrangement as our training data set.
Afghanistan_Badakhshan_to_value = [
  0.387,
   67.5
]

# scikit-learn assumes you want to predict the values for lots of regions at once, so it expects an array.
# We just want to look at a single region, so it will be the only item in our array.
regions_to_value = [
    Afghanistan_Badakhshan_to_value
]

# Run the model and make a prediction for each region in the regions_to_value
predicted_region_values = model.predict(regions_to_value)

# Since we are only predicting the percentage of one region, just look at the first prediction returned
predicted_value = predicted_region_values[0]

print("This region has an estimated value of deprivation %{:,.2f}".format(predicted_value))

