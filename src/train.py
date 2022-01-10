"""Train the data againsts multiple models."""

import argparse

from os import path

import joblib
import numpy as np
import pandas as pd

from rich.console import Console
from sklearn import metrics

import config
import model_dispatcher

console = Console()


def run(fold, model):
    """Run the fold on the selected model.

    Args:
        fold: (int) the selected fold.
        model: (str)the selected model to be trained.
    """
    df = pd.read_pickle(config.TRAINING_FOLDS_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = np.array(df_train.drop(["grade", "kfold"], axis="columns"))
    y_train = np.array(df_train.grade)

    x_valid = np.array(df_valid.drop(["grade", "kfold"], axis="columns"))
    y_valid = np.array(df_valid.grade)

    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    console.log("Fold={0}, Accuracy={1}".format(fold, accuracy))

    filepath = path.join("{0}{1}_{2}.z".format(config.MODELS, model, fold))
    joblib.dump(clf, filepath, compress=("zlib", 7))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)

    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    run(fold=args.fold, model=args.model)
