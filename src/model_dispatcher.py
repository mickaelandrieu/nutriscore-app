"""List all ML models used for this project."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

knn = KNeighborsClassifier(n_jobs=-1)
g_naive_b = GaussianNB()
d_tree_gini = DecisionTreeClassifier(criterion="gini")
d_tree_entropy = DecisionTreeClassifier(criterion="entropy")
r_forest_gini = RandomForestClassifier(criterion="gini", n_jobs=-1)
r_forest_entropy = RandomForestClassifier(criterion="entropy", n_jobs=-1)
g_boosting_deviance = GradientBoostingClassifier(loss="deviance")

models = {
    "knn": knn,
    "gaussian_naive_bayes": g_naive_b,
    "decision_tree_gini": d_tree_gini,
    "decision_tree_entropy": d_tree_entropy,
    "random_forest_gini": r_forest_gini,
    "random_forest_entropy": r_forest_entropy,
    "gradient_boosting_deviance": g_boosting_deviance,
}
