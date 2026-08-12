import pandas as pd
from scipy.sparse import csr_matrix

COLS = ["user_id", "item_id", "rating", "timestamp"]

def load_ml100k_split(data_dir, base_file="ua.base", test_file="ua.test"):
    train = pd.read_csv(f"{data_dir}/{base_file}", sep="\t", names=COLS)
    test = pd.read_csv(f"{data_dir}/{test_file}", sep="\t", names=COLS)

    user_map = {u: i for i, u in enumerate(train['user_id'].unique())}
    item_map = {i_: i for i, i_ in enumerate(train['item_id'].unique())}

    unseen_users = set(test['user_id']) - set(user_map.keys())
    unseen_items = set(test['item_id']) - set(item_map.keys())

    test_clean = test[~test['user_id'].isin(unseen_users) & ~test['item_id'].isin(unseen_items)].copy()

    train = train.copy()
    train['user_idx'] = train['user_id'].map(user_map)
    train['item_idx'] = train['item_id'].map(item_map)
    test_clean['user_idx'] = test_clean['user_id'].map(user_map)
    test_clean['item_idx'] = test_clean['item_id'].map(item_map)

    m, n = len(user_map), len(item_map)
    R_train = csr_matrix((train['rating'], (train['user_idx'], train['item_idx'])), shape=(m, n))

    return {
        "train": train, "test_clean": test_clean,
        "m": m, "n": n, "R_train": R_train,
        "u_test": test_clean['user_idx'].values,
        "i_test": test_clean['item_idx'].values,
        "y_true": test_clean['rating'].values,
    }