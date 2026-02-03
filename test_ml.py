import pytest
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_returns_correct_shapes():
    data = pd.read_csv("data/census.csv").head(200)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == 200
    assert y.shape[0] == 200


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_random_forest():
    X = [[0, 1], [1, 0], [1, 1], [0, 0]]
    y = [0, 1, 1, 0]

    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_returns_valid_range():
    y = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]

    p, r, f = compute_model_metrics(y, preds)
    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= f <= 1