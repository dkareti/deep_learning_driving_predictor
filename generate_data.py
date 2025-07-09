import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

#define an enum
BEHVIORS = ['accelerate', 'brake' 'turn_left', 'turn_right', 'cruise']

#create a dictionary
store_labels = {b : index for index, b in enumerate(BEHVIORS)}

#assuming we calculate each data point at 20 Hz for 10 sec
#####CALCULATION#######
# 1 / 20 Hz        = 0.2 sec
# 10 sec / 0.2 sec = 50 sequences
#######################
def generate_data_point(seq_len = 50):
    action = random.choice(BEHVIORS)

    if action == 'accelerate':
        accel = np.random.normal(2.0, 0.5, seq_len)
        speed = np.cumsum(accel)
        yaw_rate = np.zeros(seq_len)
        steering = np.zeros(seq_len)
        throttle = np.clip(accel / 5, 0, 1)
        brake = np.zeros(seq_len)

    elif action == 'turn_left':
        accel = np.random.normal(0.5, 0.3, seq_len)
        speed = np.cumsum(accel)
        yaw_rate = np.random.normal(20, 5, seq_len)
        steering = np.random.normal(15, 5, seq_len)
        throttle = np.clip(accel / 5, 0, 1)
        brake = np.zeros(seq_len)

    elif action == 'brake':
        accel = np.random.normal(-3.0, 0.5, seq_len)
        speed = np.clip(np.cumsum(accel[::-1]), 0, 100)[::-1]
        yaw_rate = np.zeros(seq_len)
        steering = np.zeros(seq_len)
        throttle = np.zeros(seq_len)
        brake = np.clip(-accel / 5, 0 , 1)

    elif action == 'turn_right':
        accel = np.random.normal(0.5, 0.3, seq_len)
        speed = np.cumsum(accel)
        yaw_rate = np.random.normal(-20, 5, seq_len)
        steering = np.random.normal(-15, 5, seq_len)
        throttle = np.clip(accel / 5, 0, 1)
        brake = np.zeros(seq_len)

    else: #cruise
        accel = np.random.normal(0.0, 0.1, seq_len)
        speed = np.cumsum(accel) + np.random.uniform(30, 60)
        yaw_rate = np.zeros(seq_len)
        steering = np.zeros(seq_len)
        throttle = np.full(seq_len, 0.3)
        brake = np.zeros(seq_len)

    features = np.stack([speed, accel, yaw_rate, steering, throttle, brake], axis=1)
    label = store_labels[action]

    return features, label

def create_dataset(num_sequences = 500, sequence_len=50):
    X, y = [], []

    for _ in range(num_sequences):
        #generate sequence and label using helper function
        seq, label = generate_data_point(seq_len=sequence_len)
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

#perform train, test split
#use stratify=y to force fn to balance the test set
def get_train_test_data(testing_percentage = 0.2, random_state = 42 ):
    X, y = create_dataset()
    return train_test_split(X, y, test_size=testing_percentage, random_state=random_state, stratify=y)

def print_data():
    X_train, X_test, y_train, y_test = get_train_test_data()
    print('\nThe Training Feature Matrix Shape:', X_train.shape, '\nThe Testing Feature Matrix Shape:', X_test.shape)
    print('\nThe Training Target Vector Shape:', y_train.shape, '\nThe Testing Target Vector Shape:', y_test.shape)

if __name__ == '__main__':
    print_data()