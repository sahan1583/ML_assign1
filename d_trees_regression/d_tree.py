import numpy as np
import pandas as pd
import math


class Node():
    def __init__(self, fi=None, t=None, left=None, right=None, var=None, value=None):
        self.fi = fi
        self.t = t
        self.left = left
        self.right = right
        self.var = var
        self.value = value


class Dtr():
    def __init__(self, min_s=2, max_d=2):

        self.depth = 0
        self.root = None

        self.min_s = min_s
        self.max_d = max_d

    def build_tree(self, dataset, cd=0):
        if (cd > self.depth):
            self.depth = cd
        x, y = dataset[:, :-1], dataset[:, -1]

        num_samp, num_feat = np.shape(x)
        best_split = {}
        if num_samp >= self.min_s:
            best_split = self.get_best_split(dataset, num_samp, num_feat)
            if best_split["var"] > 0:
                ls = self.build_tree(best_split["dataset_left"], cd + 1)
                # recur right
                rs = self.build_tree(best_split["dataset_right"], cd + 1)
                # return decision node
                return Node(best_split["fi"], best_split["t"],
                            ls, rs, best_split["var"])

        # compute leaf node
        lv = self.calculate_leaf_value(y)
        return Node(value=lv)

    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    def get_best_split(self, dataset, num_samp, num_feat):
        best_split = {}
        best_split["var"] = 0
        max_var = -float("inf")
        for fi in range(num_feat):
            fv = dataset[:, fi]
            pt = np.unique(fv)
            # loeop over all the feature values present in the data
            for t in pt:
                dataset_left, dataset_right = self.split(dataset, fi, t)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var = self.variance_reduction(y, left_y, right_y)
                    if curr_var > max_var:
                        best_split["fi"] = fi
                        best_split["t"] = t
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var"] = curr_var
                        max_var = curr_var

        # return best split
        return best_split

    def calculate_leaf_value(self, y):

        val = np.mean(y)
        return val

    def print_tree(self, tree=None, sc=" "):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.fi), "<=", tree.t, "?", tree.var)
            print("%sleft:" % (sc), end="")
            self.print_tree(tree.left, sc + sc)
            print("%sright:" % (sc), end="")
            self.print_tree(tree.right, sc + sc)

    def fit(self, x, y):
        dataset = np.concatenate((x, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, x):
        p = [self.make_prediction(x1, self.root) for x1 in x]
        return p

    def make_prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.fi]
        if feature_val <= tree.t:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def split(self, dataset, fi, t):
        dataset_left = np.array([row for row in dataset if row[fi] <= t])
        dataset_right = np.array([row for row in dataset if row[fi] > t])
        return dataset_left, dataset_right


if __name__ == '__main__':
    data = pd.read_csv("Train_B_Tree.csv")
    print(data)
    shuffle_df = data.sample(frac=1)
    x = shuffle_df.iloc[:, :-1].values
    y = shuffle_df.iloc[:, -1].values.reshape(-1, 1)

    #
    train_size = int(0.7 * len(x))

    x_train = x[:train_size]
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    reg = Dtr(min_s=3, max_d=25)
    reg.fit(x_train, y_train)
    print("Height of the regression tree\n")
    print(reg.depth)
    y_pred = reg.predict(x_test)

    mse = (np.sum((y - y_pred) ** 2))
    mse /= len(y)

    print("root of mean square error: ", math.sqrt(mse))






    
