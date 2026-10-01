import numpy as np

from node import Node
from decision_tree import DecisionTree
from random_forest import RandomForest


def test_gini_index():
    node = Node()
    assert node.gini_index(np.array([1, 1, 1, 1])) == 0
    assert node.gini_index(np.array([1, 1, 0, 0])) == 0.5


def test_split_threshold_separates_the_classes():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree({})
    tree.train(X, y)
    assert tree.root_node.feature_value == 2.5
    assert tree.root_node.left_child.feature_idx is None  # both children are pure leaves
    assert tree.root_node.right_child.feature_idx is None
    assert tree.predict(X) == [0, 0, 1, 1]


def test_split_with_feature_subset_uses_a_column_of_the_full_matrix():
    #column 0 is constant, so only column 1 can be split; with a subset of 1 column
    #the chosen column must be reported by its index in X, not in the subset
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    y = np.array([0, 0, 1, 1])
    for seed in range(20):
        np.random.seed(seed)
        feature, value = Node().find_best_split(X, y, feature_subset=1)
        assert feature in (None, 1)
        if feature == 1:
            assert value == 2.5


def test_bagging_samples_with_replacement():
    np.random.seed(0)
    X = np.arange(1000).reshape(-1, 1)
    X_bag, _ = RandomForest({}).bagging(X, X[:, 0])
    assert len(X_bag) == 1000
    #a bootstrap sample holds about 63% (1 - 1/e) of the distinct rows
    assert 0.55 < len(np.unique(X_bag)) / 1000 < 0.70


def test_forest_learns_a_rule_on_one_of_many_features():
    np.random.seed(0)
    X = np.random.rand(300, 6)
    y = (X[:, 4] > 0.5).astype(int)
    forest = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 5})
    forest.train(X[:200], y[:200])
    accuracy = np.mean(np.round(forest.predict(X[200:])) == y[200:])
    assert accuracy > 0.9, accuracy


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} tests passed")
