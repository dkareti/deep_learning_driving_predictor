import os
from keras.models import load_model
from src_code.eval import evaluate_model

##### ++++++++++++++++++++++++++++++++
#####
##### Checks if model exists in model path
####
#### -----------------------------------
model_path = "saved_models/behav_model.h5"

if os.path.exists(model_path):
    print('Loading the Saved Model!')
    model = load_model(model_path)
    evaluate_model(model)
else:
    evaluate_model()