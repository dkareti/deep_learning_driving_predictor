import os
from keras.models import load_model
from src_code.eval import evaluate_model
import numpy as np

##### ++++++++++++++++++++++++++++++++
#####
##### Checks if model exists in model path
####
#### -----------------------------------
model_path = "saved_models/best_model.keras"

def print_dict(dictionary: dict) -> None:
    for k, v in dictionary.items():
        print(f'{k} is: \n\n {v}')

def run_model():
    if os.path.exists(model_path):
        print('Loading the Saved Model!')
        model = load_model(model_path)

        result = evaluate_model(model)
    else:
        result = evaluate_model()

    # convert NumPy arrays to lists so Flask can serialize them
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()

    print_dict(result)

    return result