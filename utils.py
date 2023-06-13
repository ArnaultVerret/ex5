import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

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

def apply_estimator(df, X, Y, n_dt, t_max=1):
    """
    Predicts X(t) with df = V the vector field, and compares the comparaison to a given position in the futur

    args
        df (lambda t, x):
            vector field - each points' derivative in time
        X (np.ndarray):
            data points at t=0
        Y (np.ndarray):
            ground truth at t=dt (unknown dt)
        n_dt (int):
            number of samples in time (from t=0 to t=t_max)
        t_max (opt. scalar):
            prediction's duration

    returns:
        sols (np.ndarray):
            Trajectories of each data points
        mses (np.ndarray):
            mean square error for each timestamp
    """

    # predict trajectories for each data points
    sols = []
    for x in X:
        sol = solve_ivp(df, [0, t_max], x, t_eval=np.linspace(0, t_max, n_dt))
        sols.append(sol.y)

    sols = np.asarray(sols)

    # compute mean square error for each timestamp
    mses = []
    for i in range(n_dt):
        mses.append(mean_squared_error(Y, sols[:,:,i]))
    mses = np.asarray(mses)

    return sols, mses


