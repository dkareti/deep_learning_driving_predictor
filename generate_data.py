import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

#define an enum
BEHVIORS = ['accelerate', 'brake' 'turn_left', 'turn_right', 'cruise']

#create a dictionary
store_labels = {b : index for index, b in enumerate(BEHVIORS)}

#assuming we calculate each data point at 20 Hz for 10 sec
# def generate_data_point(seq_len = 50):
#     action = random.choice(BEHVIORS)

#     if action 
