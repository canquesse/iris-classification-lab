import numpy as np
import pytest
from iris_lab.experiment import split_data, run_experiment, predict_measurements, candidate_models
from sklearn.pipeline import Pipeline


def test_split_is_disjoint_reproducible_and_stratified():
    iris, train, test=split_data()
    assert not set(train) & set(test)
    assert len(train)==120 and len(test)==30
    assert np.array_equal(test,split_data()[2])
    assert np.array_equal(np.bincount(iris.target[test]),[10,10,10])


def test_scaling_is_inside_cross_validation_pipeline():
    models=candidate_models()
    assert isinstance(models["nearest_neighbors"],Pipeline)
    assert isinstance(models["logistic_regression"],Pipeline)


def test_experiment_artifacts_and_prediction(tmp_path):
    result=run_experiment(tmp_path)
    assert len(result["comparison"])==4
    assert result["baseline_test_accuracy"]==pytest.approx(1/3)
    assert sum(map(sum,result["confusion_matrix"]))==30
    assert (tmp_path/"confusion-matrix.png").stat().st_size>1000
    prediction=predict_measurements(tmp_path/"model.joblib",[5.1,3.5,1.4,.2])
    assert prediction["species"]=="setosa"
    assert sum(prediction["probabilities"].values())==pytest.approx(1)
    with pytest.raises(ValueError): predict_measurements(tmp_path/"model.joblib",[1,2,3,float("nan")])
    with pytest.raises(ValueError): predict_measurements(tmp_path/"model.joblib",[1,-2,3,4])
