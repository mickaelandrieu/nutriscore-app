"""Report performance of every model in one table."""

import argparse

from time import time

import numpy as np
import pandas as pd

from rich.console import Console
from rich.table import Table
from sklearn import metrics

import config
import model_dispatcher

console = Console()


def run(fold):
    """Run the fold on the list of models.

    Args:
        fold: (int) the selected fold.
    """
    df = pd.read_pickle(config.TRAINING_FOLDS_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = np.array(df_train.drop("grade", axis="columns"))
    y_train = np.array(df_train.grade)

    x_valid = np.array(df_valid.drop("grade", axis="columns"))
    y_valid = np.array(df_valid.grade)

    report = Table(title="Models performance of Dataset fold {0}".format(fold))
    report.add_column("Model", style="cyan")
    report.add_column("Time (s)", style="magenta")
    report.add_column("Accuracy", style="green")

    for name in model_dispatcher.models.keys():
        clf = model_dispatcher.models[name]

        start = time()
        clf.fit(x_train, y_train)
        preds = clf.predict(x_valid)
        accuracy = metrics.accuracy_score(y_valid, preds)
        elapsed_time = str(round(time() - start, 3))

        report.add_row(name, elapsed_time, str(accuracy))

    console.print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)

    args = parser.parse_args()
    run(fold=args.fold)
