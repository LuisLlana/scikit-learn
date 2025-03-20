import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_array_equal

@pytest.mark.parametrize("criterion", ["jdj"])
def test_missing_values_best_splitter_three_classes(criterion):
    """Test when missing values are uniquely present in a class among 3 classes."""
    missing_values_class = 0
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 8, 9, 11, 12]]).T
    y = np.array([missing_values_class] * 4 + [1] * 4 + [2] * 4)
    dtc = DecisionTreeClassifier(random_state=42, max_depth=2, criterion=criterion)
    dtc.fit(X, y)

    X_test = np.array([[np.nan, 3, 12]]).T
    y_nan_pred = dtc.predict(X_test)
    # Missing values necessarily are associated to the observed class.
    assert_array_equal(y_nan_pred, [missing_values_class, 1, 2])
