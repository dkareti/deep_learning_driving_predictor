from src_code.train import train_model
from src_code.generate_data import get_train_test_data
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(stored_model=None):
    if(stored_model is not None):
        model = stored_model
    else:
        model = train_model()

    #### ++++++++++++++++++++++++++++++++++
    #### Get X_train, X_test, y_train_y_test
    ###
    #### We can gather these sets because of the 
    ####    random state in the get_train_test func.
    #### This can produce reproducable results.
    #### -------------------------------------

    X_train, X_test, y_train, y_test = get_train_test_data()
    

    #make predictions
    y_pred = model.predict(X_test).argmax(axis=1)

    # Evaluate
    eval_results = model.evaluate(X_test, y_test, verbose=0)
    loss = eval_results[0] if isinstance(eval_results, (list, tuple)) else eval_results


    # Metrics
    c = confusion_matrix(y_test, y_pred)
    cl = classification_report(y_test, y_pred)

    return {
        "Confusion Matrix": c,
        "Classification Report": cl,
        "Loss": loss
    }