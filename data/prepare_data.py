import pandas as pd
import torch


def read_data(dataset):
        # Add code to read your dataset. 
        # It expects a pandas data frame called df with the response, the weighting factor and all covariates.
        # Also, create the strings response_name and weights_name with the exact names in df corresponding to these values


    return df, response_name, weights_name


def scale_data(df, weights_name, scale='max'):
    weights = df[weights_name]
    df = df.drop(weights_name, axis=1)
    weights /= weights.max()
    weights = weights.values
    if scale == 'max':
        max_values = df.max()
        df /= max_values
    else:
        raise NotImplementedError
    return df, max_values, weights


def data_pytorch(df, response_name):
    # Select train objects
    y = df[response_name]
    X = df.drop(response_name, axis=1)

    # Optimisation requires plain values
    X = X.values
    y = y.values
    dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=X.shape[0])
    return data_loader, X, y
