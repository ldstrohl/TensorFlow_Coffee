# Coffee Experiment
import tensorflow as tf
import pandas as pd
from data_read import data_read
# Data read
#   Call data read function to import and parse spreadsheet
#   Should result in a pandas dataframe with parameter/feature definitions
filename = "CoffeeData"
data = data_read(filename)

# 
