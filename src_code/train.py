from src_code.generate_data import get_train_test_data
from src_code.model import build_model

###### +++++++++++++++++++++++++++++++++++++++++++++
#### Create a TRAINING FUNCTION
#### -----------------------------------------------
#### 
#### Generate the test and train data, build the model, and fit it accordingly
####
#### -----------------------------------------------

def train_model():
    X_train, X_test, y_train, y_test = get_train_test_data()

    #### ++++++++++++++++++++++++++++++++++++++++++
    #### Shapes
    ####
    #### The shapes are as follows: 
    ####    1. X_train:
    ####        (400, 50, 6) 
    ####    2. X_test:
    ####        (100, 50, 6) 
    ####    3. y_train:
    ####        (400,) 
    ####    4. y_test:
    ####        (100,)
    #### -------------------------------------------

    model = build_model(input_shape=X_train.shape[1:], num_classes=5 )

    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save("saved_models/behav_model.keras")
    return model