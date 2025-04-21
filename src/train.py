"""Train the data againsts multiple models."""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from rich.console import Console
from sklearn import metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config
import model_dispatcher

console = Console()

def run(fold, model):
    """Run the fold on the selected model.

    Args:
        fold: (int) the selected fold.
        model: (str) the selected model to be trained.
    """
    # Vérification des chemins
    if not os.path.exists(config.TRAINING_FOLDS_FILE):
        raise FileNotFoundError(f"Le fichier {config.TRAINING_FOLDS_FILE} n'existe pas")
    
    # Création des dossiers nécessaires
    Path(config.MODELS).mkdir(parents=True, exist_ok=True)
    Path(config.DOCS).mkdir(parents=True, exist_ok=True)

    # Chargement des données
    console.print(f"[bold blue]Chargement des données depuis {config.TRAINING_FOLDS_FILE}[/bold blue]")
    df = pd.read_pickle(config.TRAINING_FOLDS_FILE)

    # Préparation des données
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Séparation des features et de la target
    x_train = df_train.drop(["grade", "kfold"], axis="columns")
    y_train = df_train.grade
    x_valid = df_valid.drop(["grade", "kfold"], axis="columns")
    y_valid = df_valid.grade

    # Normalisation des données
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    # Sélection et entraînement du modèle
    console.print(f"[bold green]Entraînement du modèle {model}[/bold green]")
    clf = model_dispatcher.models[model]

    # Validation croisée
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, x_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    console.print(f"[bold]Scores de validation croisée:[/bold] {cv_scores}")
    console.print(f"[bold]Score moyen:[/bold] {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Entraînement final
    with console.status("[bold green]Entraînement en cours...[/bold green]"):
        clf.fit(x_train_scaled, y_train)

    # Évaluation
    preds = clf.predict(x_valid_scaled)
    accuracy = metrics.accuracy_score(y_valid, preds)
    console.print(f"[bold]Fold={fold}, Accuracy={accuracy:.4f}[/bold]")

    # Métriques détaillées
    console.print("\n[bold]Rapport de classification:[/bold]")
    console.print(metrics.classification_report(y_valid, preds))

    # Matrice de confusion
    cm = metrics.confusion_matrix(y_valid, preds)
    console.print("\n[bold]Matrice de confusion:[/bold]")
    console.print(cm)

    # Sauvegarde du modèle et du scaler
    model_path = os.path.join(config.MODELS, f"{model}_{fold}.z")
    scaler_path = os.path.join(config.MODELS, f"scaler_{model}_{fold}.z")
    
    joblib.dump(clf, model_path, compress=("zlib", 7))
    joblib.dump(scaler, scaler_path, compress=("zlib", 7))
    
    console.print(f"\n[bold green]Modèle sauvegardé dans {model_path}[/bold green]")
    console.print(f"[bold green]Scaler sauvegardé dans {scaler_path}[/bold green]")

    # Sauvegarde des features importantes si le modèle en a
    if hasattr(clf, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': x_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(
            os.path.join(config.DOCS, f"feature_importance_{model}_{fold}.csv"),
            index=False
        )
        console.print("\n[bold]Importance des features:[/bold]")
        console.print(feature_importance.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement des modèles de prédiction Nutriscore")
    parser.add_argument("--fold", type=int, required=True, help="Numéro du fold à utiliser")
    parser.add_argument("--model", type=str, required=True, choices=list(model_dispatcher.models.keys()),
                       help="Modèle à entraîner")
    args = parser.parse_args()
    
    try:
        run(fold=args.fold, model=args.model)
    except Exception as e:
        console.print(f"[bold red]Erreur: {str(e)}[/bold red]")
        raise
