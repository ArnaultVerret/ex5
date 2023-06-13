import numpy as np

def load_data(str_file):
    """
    Load data from txt file into a numpy array

    args:
        str_file: if 0 or 1, load known file for the exercise
                  if str, load specific file

    returns:
        d (np.ndarray): of shape (n_data, space_dim) - loaded data
    """
    if str_file == 0:
        d = np.loadtxt("nonlinear_vectorfield_data_x0.txt")
    elif str_file == 1:
        d = np.loadtxt("nonlinear_vectorfield_data_x1.txt")
    else:
        d = np.loadtxt(str_file)
    return d




