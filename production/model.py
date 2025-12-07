import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str)
    args = parser.parse_args()

    data = pd.read_csv(args.training_data)
    print(data)

   # Separate features and target
    target_col = "fraud_reported"  # change if your label column differs
    X = data.drop(columns=[target_col], axis=1)
    y = data[target_col]

 # Train-test split
    # splitting data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]
    
    # Scaling the numeric values in the dataset


    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_df)
    scaled_num_df = pd.DataFrame(data = scaled_data, columns = num_df.columns, index = X_train.index)
    X_train.drop(columns = scaled_num_df.columns, inplace = True)
    X_train = pd.concat([scaled_num_df, X_train], axis = 1)

    #Random Forest Model
# Define Random Forest model
    rand_clf = RandomForestClassifier(criterion= 'entropy', max_depth= 10, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 140)
    rand_clf.fit(X_train, y_train)

    y_pred = rand_clf.predict(X_test)
    acc = np.average(y_pred == y_test)
    print("Accuracy", acc)

    # accuracy_score, confusion_matrix and classification_report

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    svc_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))
    svc_test_acc = accuracy_score(y_test, y_pred)

    print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
    print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Classification metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)

    # Log main metrics
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1", report["weighted avg"]["f1-score"])

    print("precision", report["weighted avg"]["precision"])
    print("recall", report["weighted avg"]["recall"])
    print("f1", report["weighted avg"]["f1-score"])

# Log the model in MLflow
# Add these lines right here to log the model
    print("Logging the model with MLflow...")
    mlflow.sklearn.log_model(
        sk_model=rand_clf,
        artifact_path="claimfraud_rf",
        registered_model_name="claimsfraud_rf_model"
    )
print("Model logged successfully.")


if __name__ == "__main__":
    main()
