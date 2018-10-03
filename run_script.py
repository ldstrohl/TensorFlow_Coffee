# Coffee Experiment: Optimization of a cup of pour-over coffee
# run_script.py
# Lloyd Strohl III
# 09/28/18

import tensorflow as tf
import pandas as pd
import numpy as np
from data_read import data_read


# Data read
#   Read in data and shuffle to avoid pathological ordering

filename = "CoffeeExperimentData.csv"
coffee_data = data_read(filename)
coffee_data = coffee_data.reindex(np.random.permutation(coffee_data.index))

# Build model
# Define the input feature: grounds mass
feature_name = "Grounds mass (g)"
my_feature = coffee_data[[feature_name]]
# Configure a numeric feature column for grounds mass
feature_columns = [tf.feature_column.numeric_column(feature_name)]
# define the label
label_name = "Cup Rating (0 to 10)"
targets = coffee_data[label_name]

# Configure the LinearRegressor
# Use gradient descent as the optimizer for training the model
# set a learning rate of 0.0000001 for GD
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns, optimizer=my_optimizer
)
