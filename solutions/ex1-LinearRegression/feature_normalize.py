def feature_normalize(X, mu=None, sigma=None):
    # Calculate the means (column by column):
    if mu is None:
        mu = X.mean(axis=0)

    # Calculate the standard deviations (column by column).
    if sigma is None:
        sigma = X.std(axis=0)

    # Subtract the means from all the columns, then divide by the standard deviation:
    X = (X - mu) / sigma

    return X, mu, sigma
