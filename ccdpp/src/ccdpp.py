# Serial numpy version of CCD++ algorithm (not parallelized)
import numpy as np
from scipy.sparse import csr_matrix


def train_ccdpp_es(R_train, k, lam, T=5, max_iter=30, patience=2,
                   val_data=None, seed=42, verbose=False):
    """
    CCD++ serial version (Algorithm 2, Yu et al. 2014), same paper notation:
    W (m x k), H (n x k), R = observed residuals, w_t/h_t = t-th latent column.
    Training with early stopping.
    Used for hyperparameter tuning (k, lam) with validation set and
    for finding the optimal number of iterations.
    """
    m, n = R_train.shape
    rng = np.random.default_rng(seed)

    W = np.zeros((m, k))  # W = 0 at the beginning (algorithm 2, line 1)
    H = rng.normal(scale=0.01, size=(n, k))

    R = R_train.tocsr().astype(float)  # R = A at the beginning
    pattern = R.copy(); pattern.data[:] = 1.0  # binary mask, where we have observed ratings
    pattern_T = pattern.T.tocsr()  # transpose of the pattern matrix

    rows, cols = R.tocoo().row, R.tocoo().col  # indices of observed ratings

    best_rmse, best_W, best_H, no_improve = np.inf, None, None, 0

    for outer in range(max_iter):
        for t in range(k):
            w_t, h_t = W[:, t], H[:, t]  # t-th latent column vectors

            # Build R_hat (eq.16)
            R_hat = R.copy()
            R_hat.data = R.data + w_t[rows] * h_t[cols]
            R_hat_T = R_hat.T.tocsr()

            u, v = w_t.copy(), h_t.copy()  # copy of the latent vectors for updates

            # T iterations of coordinate descent for subproblem rank-1 (eq.17)
            for _ in range(T):
                # Update u (eq.18): row-wise, sum_j(R_hat_ij v_j) / (lambda + sum_j v_j^2)
                u = R_hat.dot(v) / (lam + pattern.dot(v ** 2))
                # Update v (eq.19): same but colmun-wise
                v = R_hat_T.dot(u) / (lam + pattern_T.dot(u ** 2))

            # Update of (w_t, h_t) and R (eq. 20-21)
            W[:, t], H[:, t] = u, v
            R.data = R_hat.data - u[rows] * v[cols]

        u_val, i_val, y_val = val_data
        rmse_val = np.sqrt(np.mean((y_val - predict(W, H, u_val, i_val)) ** 2))
        if verbose:
           print(f"  iter {outer+1}: val RMSE {rmse_val:.4f}")

        if rmse_val < best_rmse - 1e-4:
            best_rmse, best_W, best_H, best_iter, no_improve = rmse_val, W.copy(), H.copy(), outer + 1, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_W, best_H, best_rmse , best_iter


def train_ccdpp_fixed(R_train, k, lam, n_iter, T=5, seed=42):
    """
    CCD++ training with a fixed number of iterations.
    FInal version used for training the complete dataset with optimized parmeters.
    """
    m, n = R_train.shape
    rng = np.random.default_rng(seed)
    W = np.zeros((m, k))
    H = rng.normal(scale=0.01, size=(n, k))
    R = R_train.tocsr().astype(float)
    pattern = R.copy(); pattern.data[:] = 1.0
    pattern_T = pattern.T.tocsr()
    rows, cols = R.tocoo().row, R.tocoo().col

    for outer in range(n_iter):
        for t in range(k):
            w_t, h_t = W[:, t], H[:, t]
            R_hat = R.copy()
            R_hat.data = R.data + w_t[rows] * h_t[cols]
            R_hat_T = R_hat.T.tocsr()
            u, v = w_t.copy(), h_t.copy()
            for _ in range(T):
                u = R_hat.dot(v) / (lam + pattern.dot(v ** 2))
                v = R_hat_T.dot(u) / (lam + pattern_T.dot(u ** 2))
            W[:, t], H[:, t] = u, v
            R.data = R_hat.data - u[rows] * v[cols]

    return W, H


def predict(W, H, user_idx, item_idx):
    return np.sum(W[user_idx] * H[item_idx], axis=1)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))