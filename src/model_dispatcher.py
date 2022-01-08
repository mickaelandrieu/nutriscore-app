"""List all ML models used for this project."""
from sklearn.tree import DecisionTreeClassifier

models = {
    "decision_tree_gini": DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": DecisionTreeClassifier(criterion="entropy"),
}
