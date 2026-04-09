from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def evaluate_model(results):

    y_test = results["y_test"]
    y_pred = results["y_pred"]

    print("\n=== Evaluation ===")

    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
