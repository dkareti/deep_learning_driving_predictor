from src_code.train import train_model
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model():
    X_test, y_test, model = train_model()
    

    #make predictions
    y_pred = model.predict(X_test).argmax(axis=1)

    #metrics
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    loss = model.evaluate(X_test, y_test)
    print(f'\n\n Loss: {loss}')