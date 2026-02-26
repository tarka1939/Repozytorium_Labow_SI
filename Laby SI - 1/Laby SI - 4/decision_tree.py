from collections import defaultdict
import numpy as np
from node import Node

class DecisionTree:
    def __init__(self, params):
        self.root_node = Node()
        self.params = defaultdict(lambda: None, params)


    def train(self, X, y):
        self.root_node.train(X, y, self.params)
        self.root_node.feature_subset = self.params["feature_subset"]
        self.root_node.depth = self.params["depth"]
        if self.root_node.feature_subset is None:
            self.root_node.feature_subset = X.shape[1]
            if self.root_node.depth is None:
                self.root_node.depth = 1000
            else:
                self.root_node.feature_subset = min(self.root_node.feature_subset, X.shape[1])
                    

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X):
        prediction = []
        for x in X:
            prediction.append(self.root_node.predict(x))
        return prediction

