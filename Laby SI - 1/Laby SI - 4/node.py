import copy

import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None
    
    def gini_index(self, y):
        # y - tablica etykiet (0 lub 1)
        if len(y) == 0:
            return 0

        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        total = pos + neg
        p_pos = pos / total
        p_neg = neg / total
        #return p_pos**2 + p_neg**2
        return 1 - p_pos**2 - p_neg**2

    def gini_gain(self, y, left_idx):
        # y - tablica etykiet
        # left_idx - indeks podziału (int)
        left = y[:left_idx]
        right = y[left_idx:]
        n = len(y)
        n_left = len(left)
        n_right = len(right)
        if n_left == 0 or n_right == 0:
            return 0
        gini_left = self.gini_index(left)
        gini_right = self.gini_index(right)
        #parent_gini = self.gini_index(y)
        gain = 1 - (n_left / n) * gini_left - (n_right / n) * gini_right
        return gain

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0

        # TODO find position of best data split
        for idx in range(len(possible_splits)):
            # split point i puts elements 0..i on the left (the threshold is the midpoint of i and i+1),
            # so the left part is y[:i+1]
            gain = self.gini_gain(y, possible_splits[idx] + 1)
            if gain > best_gain:
                best_gain = gain
                best_idx = possible_splits[idx]
        
        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None

        # TODO implement feature selection
        if feature_subset is not None:
            if feature_subset > X.shape[1]:
                feature_subset = X.shape[1]
            features = np.random.choice(
                X.shape[1], feature_subset, replace=False)
            X = X[:, features]
        else:
            features = np.arange(X.shape[1])
                

        for d in range(X.shape[1]):
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, order[[idx, idx + 1]])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        # d indexes the selected subset of columns; return the index of the column in the full X
        return features[best_split[0]], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
