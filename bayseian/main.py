import numpy as np
import pandas as pd

import math


def func(gender):
    if gender == "Male":
        return 2
    return 1


def calculate_prior_l(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append((len(df[df[Y] == i]) + 2) / (len(df) + 4))
    return prior


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y] == i]) / len(df))
    return prior


def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y] == label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val - mean) ** 2 / (2 * std ** 2)))
    return p_x_given_y


def naive_bayes_gaussian(df, X, Y, var=False):
    # get feature names
    features = list(df.columns)[:-1]
    prior = 0
    if var:
        prior = calculate_prior_l(df, Y)
    # calculate prior
    else:
        prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)


if __name__ == '__main__':
    data = pd.read_csv("Train_B_Bayesian.csv")
    data["gender"] = data["gender"].apply(func)
    data
    x1 = data.iloc[:, 0:1].values
    x2 = data.iloc[:, 2:3].values
    x3 = data.iloc[:, 3:4].values
    x4 = data.iloc[:, 4:5].values
    x5 = data.iloc[:, 5:6].values
    x6 = data.iloc[:, 6:7].values
    x7 = data.iloc[:, 7:8].values
    x8 = data.iloc[:, 8:9].values
    x9 = data.iloc[:, 9:10].values
    ar = []
    ar.append(2 * np.mean(x1))
    ar.append(2 * np.mean(x2))
    ar.append(2 * np.mean(x3))
    ar.append(2 * np.mean(x4))
    ar.append(2 * np.mean(x5))
    ar.append(2 * np.mean(x6))
    ar.append(2 * np.mean(x7))
    ar.append(2 * np.mean(x8))
    ar.append(2 * np.mean(x9))
    ar[0] = ar[0] + 5 * np.std(x1)
    ar[1] = ar[1] + 5 * np.std(x2)
    ar[2] = ar[2] + 5 * np.std(x3)
    ar[3] = ar[3] + 5 * np.std(x4)
    ar[4] = ar[4] + 5 * np.std(x5)
    ar[5] = ar[5] + 5 * np.std(x6)
    ar[6] = ar[6] + 5 * np.std(x7)
    ar[7] = ar[7] + 5 * np.std(x8)
    ar[8] = ar[8] + 5 * np.std(x9)
    b = []
    for ind in range(0, 583):
        c = 0
    if data["age"][ind] > ar[0]:
        c = c + 1
    if data["tot_bilirubin"][ind] > ar[1]:
        c = c + 1
    if data["direct_bilirubin"][ind] > ar[2]:
        c = c + 1
    if data["tot_proteins"][ind] > ar[3]:
        c = c + 1
    if data["albumin"][ind] > ar[4]:
        c = c + 1
    if data["ag_ratio"][ind] > ar[5]:
        c = c + 1
    if data["sgpt"][ind] > ar[6]:
        c = c + 1
    if data["sgot"][ind] > ar[7]:
        c = c + 1
    if data["alkphos"][ind] > ar[8]:
        c = c + 1
    if c >= 2:
        b.append(ind)


    print(data)
    for c in data.columns:
        data[c] = (data[c] - data[c].min()) / (data[c].max() - data[c].min())
    damta = data.sample(frac=0.7)
    l = int(0.7 * (data.shape[0] - 1))
    temst = data[l:]
    m = []
    import math

    p1 = math.floor(0.2 * (damta.shape[0] - 1))
    p2 = math.floor(0.4 * (damta.shape[0] - 1))
    p3 = math.floor(0.6 * (damta.shape[0] - 1))
    p4 = math.floor(0.8 * (damta.shape[0] - 1))
    s = []
    s.append(damta[:p1])
    s.append(damta[p1:p2])
    s.append(damta[p2:p3])
    s.append(damta[p3:p4])
    s.append(damta[p4:])
    max = 0
    i = 0
    max_train = 0

    for j in s:
        test = j
    ct = damta.drop(j.index)
    X_test = test.iloc[:, :-1].values
    Y_test = test.iloc[:, -1].values
    Y_pred = naive_bayes_gaussian(ct, X=X_test, Y="is_patient")
    pred_score = np.sum(Y_test == Y_pred)
    k = pred_score / len(Y_test)
    if max < k:
        max = k
        max_train = ct
    X_test = temst.iloc[:, :-1].values
    Y_test = temst.iloc[:, -1].values
    Y_pred = naive_bayes_gaussian(max_train, X=X_test, Y="is_patient")
    pred_score = np.sum(Y_test == Y_pred)
    print("accuracy percentage\n")
    print(pred_score / (len(Y_test)))
    Y_pred = naive_bayes_gaussian(max_train, X=X_test, Y="is_patient", var=True)
    pred_score = np.sum(Y_test == Y_pred)
    print("accuracy percentage after laplace")
    print(pred_score / (len(Y_test)))
