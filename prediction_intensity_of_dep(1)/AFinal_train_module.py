import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
import numpy as np

# Load the data set

df = pd.read_csv("Intensity_of_deprivation_Regional.csv")


# Remove the unecessary features

del df['MPI National']

#
features_df = pd.get_dummies(df)

# Remove the Intensity of deprivation Regional from the feature data,it represent y 
del features_df['Intensity of deprivation Regional']

# Create the X and y arrays
X = features_df.as_matrix()
y = df['Intensity of deprivation Regional'].as_matrix()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=5,
    max_features=1,
    loss='ls',
    random_state=0
)
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_Intensity of deprivation Regional_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)







