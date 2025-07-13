from keras.models import Sequential
import keras
#### +++++++++++++++++++++++++++++++++++
#### Create the model (Tweak it as necessary to acheive the best results)
####
#### The function build_model has two parameters: 
#### 1. The input shape
####    (time_steps, num_features)
#### 2. The num classes
####    # of possible output behaviors 
####   
#### -----------------------------------

def build_model(input_shape, num_classes):
    model = Sequential([
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    #compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
