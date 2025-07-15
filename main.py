import os
from keras.models import load_model
from src_code.eval import evaluate_model

##### ++++++++++++++++++++++++++++++++
#####
##### Checks if model exists in model path
####
#### -----------------------------------
model_path = "saved_models/behav_model.keras"

if os.path.exists(model_path):
    print('Loading the Saved Model!')
    model = load_model(model_path)

    result = evaluate_model(model)
    for k, v in result.items():
        print(f'{k} is: \n\n {v}\n')
else:
    result = evaluate_model()
    for k, v in result.items():
        print(f'{k} is: \n\n {v}\n')