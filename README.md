# deep_learning_driving_predictor
This repo uses neural networks to predict the driving behavior given a series of characteristics about the driving of the vehicle

# 🚗 Driving Behavior Prediction using CNN + LSTM

This project uses synthetic telemetry data to predict driver behaviors (e.g., accelerate, brake, turn) using a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) architecture.

## 📦 Project Structure

## 🧠 Problem Statement

Given a sequence of telemetry readings from a single car (speed, acceleration, steering angle, etc.), predict the driving behavior during that sequence:

- `accelerate`
- `brake`
- `turn_left`
- `turn_right`
- `cruise`

## 🧪 Data

Synthetic time-series data is generated to simulate realistic driving behavior. Each sequence:
- Has 50 time steps
- Contains 6 features per time step
- Is labeled with one driving behavior

## 🧠 Model

The model combines:
- `Conv1D` layers to learn short-term signal patterns
- `MaxPooling1D` to reduce sequence length
- `LSTM` to model long-term temporal dependencies
- `Dense` layers for final classification

## 📈 Evaluation

- Accuracy, F1 score, and confusion matrix are printed
- Evaluation is done using a test set (20%) held out from training

## 🚀 How to Run

1. Install dependencies:

`pip install -r requirements.txt`
`python3 main.py`