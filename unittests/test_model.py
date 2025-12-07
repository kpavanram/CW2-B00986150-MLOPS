import pytest
from unittest import mock
import pandas as pd
import numpy as np

import production.model  # assuming your main code is in train.py

@pytest.fixture
def fake_data():
    # Create a small synthetic dataset
    data = pd.DataFrame({
        "months_as_customer": [12, 24, 36],
        "policy_deductable": [1000, 500, 1500],
        "umbrella_limit": [0, 1000000, 0],
        "capital-gains": [0, 1000, 0],
        "capital-loss": [0, 0, 200],
        "incident_hour_of_the_day": [14, 9, 20],
        "number_of_vehicles_involved": [1, 2, 1],
        "bodily_injuries": [0, 1, 0],
        "witnesses": [0, 1, 0],
        "injury_claim": [0, 5000, 0],
        "property_claim": [0, 2000, 0],
        "vehicle_claim": [0, 3000, 0],
        "other_feature": [1,2,3],
        "fraud_reported": [0,1,0]
    })
    return data

def test_main_runs(fake_data):
    # Mock argparse to return a dummy training_data path
    with mock.patch("argparse.ArgumentParser.parse_args") as mock_args:
        mock_args.return_value.training_data = "fake_path.csv"

        # Mock pd.read_csv to return fake_data
        with mock.patch("pandas.read_csv", return_value=fake_data):
            
            # Mock MLflow methods
            with mock.patch("mlflow.sklearn.log_model") as mock_log_model, \
                 mock.patch("mlflow.log_metric") as mock_log_metric:

                # Run main
                production.model.main()

                # Assert that MLflow methods were called
                assert mock_log_model.called, "MLflow log_model was not called"
                assert mock_log_metric.called, "MLflow log_metric was not called"

def test_scaling_and_features(fake_data):
    # Test scaling part separately
    from sklearn.preprocessing import StandardScaler
    
    X = fake_data.drop(columns=["fraud_reported"])
    num_df = X[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)
    
    # Check shape
    assert scaled.shape == num_df.shape
    # Check that scaling changes mean approximately 0
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-7)
