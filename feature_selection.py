from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def select_features(X, y, method="mutual_info", k=40):

    if method == "anova":
        selector = SelectKBest(score_func=f_classif, k=k)

    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)

    else:
        raise ValueError("Invalid feature selection method")

    X_selected = selector.fit_transform(X, y)

    return X_selected
