"""Create KFold stratified dataset for final training."""
import numpy as np
import pandas as pd

from sklearn import model_selection

import config

if __name__ == "__main__":
    df = pd.read_pickle(config.TRAINING_FILE)

    df["kfold"] = -1

    # Randomize the order of appearance of data to mitigate insertion bias
    df = df.sample(frac=1).reset_index(drop=True)
    target = np.array(df.grade)

    kf = model_selection.StratifiedKFold(n_splits=5)

    splits = kf.split(X=df, y=target)
    for fold, (_train_index, test_index) in enumerate(splits):
        df.loc[test_index, "kfold"] = fold

    df.to_pickle("{0}training_folds.pkl".format(config.INPUT))
