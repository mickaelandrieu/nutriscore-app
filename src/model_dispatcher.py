"""List all ML models used for this project."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    "knn": KNeighborsClassifier(),
    "gaussian_naive_bayes": GaussianNB(),
    "decision_tree_gini": DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": DecisionTreeClassifier(criterion="entropy"),
    "random_forest_gini": RandomForestClassifier(criterion="gini"),
    "random_forest_entropy": RandomForestClassifier(criterion="entropy"),
    "gradient_boosting_deviance": GradientBoostingClassifier(loss="deviance"),
}
