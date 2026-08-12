# Spark distributed version of the CCD++ algorithm (Algorithm 3, Yu et al. 2014)
import numpy as np
from pyspark import SparkContext

NUM_PARTITIONS = 8  # for ml-100k, 2-4 partition is better

def build_grouped(rdd, group_by_user, num_partitions):
    """
    Group ratings by user (R_rows) or by item (R_cols) for distributed processing.
    If group_by_user=True: key=user_idx, value=(item_idx array, rating_array) -> R_rows
    If group_by_user=False: key=item_idx, value=(user_idx array, rating_array) -> R_cols
    """
    if group_by_user:
        kv = rdd.map(lambda r: (r[0], (r[1], r[2])))
    else:
        kv = rdd.map(lambda r: (r[1], (r[0], r[2])))

    grouped = kv.groupByKey(numPartitions=num_partitions).mapValues(
        lambda vals: (
            np.array([x[0] for x in vals], dtype=np.int32),
            np.array([x[1] for x in vals], dtype=np.float64)
        )
    )
    return grouped.persist()


def build_r_hat_rows(R_rows, w_bc, h_bc):
    """Eq. (16) / Eq. (25) for residual updates, seen by user rows: R_hat_ij = R_ij + w_t[i]*h_t[j]"""
    return R_rows.map(
        lambda kv: (kv[0], (kv[1][0], kv[1][1] + w_bc.value[kv[0]] * h_bc.value[kv[1][0]])),
        preservesPartitioning=True
    )


def build_r_hat_cols(R_cols, w_bc, h_bc):
    """Eq. (16) / Eq. (25) for residual updates, seen by item columns: R_hat_ij = R_ij + w_t[i]*h_t[j]"""
    return R_cols.map(
        lambda kv: (kv[0], (kv[1][0], kv[1][1] + w_bc.value[kv[1][0]] * h_bc.value[kv[0]])),
        preservesPartitioning=True
    )


def compute_u(R_hat_rows, v_bc, lam, m):
    """Eq. (18) / Eq. (26): u_i = sum_j(R_hat_ij * v_j)/(lam + sum_j v_j^2), for each row i."""
    def calc(kv):
        i, (items, r_hat) = kv
        vj = v_bc.value[items]
        return (i, np.dot(r_hat, vj) / (lam + np.dot(vj, vj)))
    results = R_hat_rows.map(calc).collect()
    u = np.zeros(m)
    for i, val in results:
        u[i] = val
    return u


def compute_v(R_hat_cols, u_bc, lam, n):
    """Eq. (19) / Eq. (27): v_j = sum_i(R_hat_ij * u_i)/(lam + sum_i u_i^2), for each column j."""
    def calc(kv):
        j, (users, r_hat) = kv
        ui = u_bc.value[users]
        return (j, np.dot(r_hat, ui) / (lam + np.dot(ui, ui)))
    results = R_hat_cols.map(calc).collect()
    v = np.zeros(n)
    for j, val in results:
        v[j] = val
    return v


def update_residual_rows(R_hat_rows, u_bc, v_bc):
    """Eq. (21) / Eq. (29): updates residuals for row-partitioned data."""
    return R_hat_rows.map(
        lambda kv: (kv[0], (kv[1][0], kv[1][1] - u_bc.value[kv[0]] * v_bc.value[kv[1][0]])),
        preservesPartitioning=True
    )

def update_residual_cols(R_hat_cols, u_bc, v_bc):
    """Eq. (21) / Eq. (29): updates residuals for column-partitioned data."""
    return R_hat_cols.map(
        lambda kv: (kv[0], (kv[1][0], kv[1][1] - u_bc.value[kv[1][0]] * v_bc.value[kv[0]])),
        preservesPartitioning=True
    )


def train_ccdpp_spark(sc, R_rows, R_cols, m, n, k, lam, n_iter, T=5, seed=42, checkpoint_every=5, verbose=False):
    """
    Main Spark training loop for CCD++ using RDD transformations,
    broadcast variables, and periodic checkpointing to control lineage depth.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((m, k))
    H = rng.normal(scale=0.01, size=(n, k))
    step_counter = 0

    for outer in range(n_iter):
        for t in range(k):
            w_t, h_t = W[:, t], H[:, t]
            w_bc, h_bc = sc.broadcast(w_t), sc.broadcast(h_t)

            R_hat_rows = build_r_hat_rows(R_rows, w_bc, h_bc).persist()
            R_hat_cols = build_r_hat_cols(R_cols, w_bc, h_bc).persist()

            u, v = w_t.copy(), h_t.copy()
            for _ in range(T):
                v_bc = sc.broadcast(v)
                u = compute_u(R_hat_rows, v_bc, lam, m)
                v_bc.unpersist()

                u_bc = sc.broadcast(u)
                v = compute_v(R_hat_cols, u_bc, lam, n)
                u_bc.unpersist()

            W[:, t], H[:, t] = u, v

            u_bc_f, v_bc_f = sc.broadcast(u), sc.broadcast(v)
            R_rows = update_residual_rows(R_hat_rows, u_bc_f, v_bc_f).persist()
            R_cols = update_residual_cols(R_hat_cols, u_bc_f, v_bc_f).persist()

            step_counter += 1
            if step_counter % checkpoint_every == 0:
                R_rows.localCheckpoint();
                R_cols.localCheckpoint();

            R_rows.count(); R_cols.count()

            R_hat_rows.unpersist(); R_hat_cols.unpersist()
            w_bc.unpersist(); h_bc.unpersist(); u_bc_f.unpersist(); v_bc_f.unpersist()

        if verbose:
            train_rmse = np.sqrt(
                R_rows.map(lambda kv: np.sum((kv[1][1]) ** 2)).sum() / 
                R_rows.map(lambda kv: len(kv[1][1])).sum()
            )
            print(f"iter {outer+1}/{n_iter}, train RMSE: {train_rmse:.4f}")

    return W, H