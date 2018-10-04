# Coffee Experiment: Optimization of a cup of pour-over coffee
# run_script.py
# Lloyd Strohl III
# 09/28/18

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from data_read import data_read

# Data read
#   Read in data and shuffle to avoid pathological ordering

filename = "CoffeeExperimentData.csv"
coffee_data = data_read(filename)
coffee_data = coffee_data.reindex(np.random.permutation(coffee_data.index))

# Build model
# Define the input feature: grounds mass
feature_name = "Grounds_mass_g"
my_feature = coffee_data[[feature_name]]
# Configure a numeric feature column for grounds mass
feature_columns = [tf.feature_column.numeric_column(feature_name)]
# define the label
label_name = "Cup_Rating_0to10"
targets = coffee_data[label_name]

# Configure the LinearRegressor
# Use gradient descent as the optimizer for training the model
# set a learning rate of 0.0000001 for GD
# Hyper Parameters
learning_rate = 0.0000001
gradient_norm_limit = 5.0
batch_size = 1
shuffle = True
num_epochs = None

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,
                                                           gradient_norm_limit)
# Configure the linear regression model with our feature columns and optimizer.
linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns, optimizer=my_optimizer
)


# define input function
def my_input_fn(features, targets, batch_size=batch_size, shuffle=shuffle,
                num_epochs=num_epochs):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# train the model
_ = linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=100
)

# Evaluate
prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1,
                                          shuffle=False)
# call predict() on regressor to make predictions
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([item['predictions'][0] for item in predictions])

# print MSE and RMSE
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f"
      % root_mean_squared_error)
