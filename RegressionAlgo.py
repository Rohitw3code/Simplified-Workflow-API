from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor, BayesianRidge, PassiveAggressiveRegressor, TheilSenRegressor, QuantileRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import PoissonRegressor

regression_algorithms = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'ElasticNet': ElasticNet(),
    'Huber Regressor': HuberRegressor(),
    'Bayesian Ridge Regression': BayesianRidge(),
    'Passive Aggressive Regressor': PassiveAggressiveRegressor(),
    'Theil-Sen Regressor': TheilSenRegressor(),
    'Quantile Regression': QuantileRegressor(),
    'SVR': SVR(),
    'Decision Trees Regression': DecisionTreeRegressor(),
    'K-Nearest Neighbors (KNN) Regression': KNeighborsRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'Gaussian Process Regression': GaussianProcessRegressor(),
    'Isotonic Regression': IsotonicRegression(),
    'RANSAC (RANdom SAmple Consensus)': RANSACRegressor(),
    'Polynomial Regression': PolynomialFeatures(),
    'Multi-Output Regression': MultiOutputRegressor(LinearRegression()),
    'Poisson Regression': PoissonRegressor()
}


