from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

classification_algorithms = {
    'Logistic Regression': LogisticRegression(),
    'Stochastic Gradient Descent Classifier': SGDClassifier(),
    'Perceptron': Perceptron(),
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(),
    'Support Vector Classifier (SVC)': SVC(),
    'Linear Support Vector Classifier (LinearSVC)': LinearSVC(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
    'AdaBoost Classifier': AdaBoostClassifier(),
    'Bagging Classifier': BaggingClassifier(),
    'Extra Trees Classifier': ExtraTreesClassifier(),
    'K-Nearest Neighbors (KNN) Classifier': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Complement Naive Bayes': ComplementNB(),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis (QDA)': QuadraticDiscriminantAnalysis(),
    'Multi-layer Perceptron Classifier (Neural Network)': MLPClassifier()
}