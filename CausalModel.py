import dowhy
from dowhy import CausalModel
import dowhy.datasets

from econml.metalearners import XLearner, SLearner, TLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor

def dowhyCausal():
    
    est = TLearner(models=GradientBoostingRegressor())
    est.fit(Y, T, X=np.hstack([X, W]))
    treatment_effects = est.effect(np.hstack([X_test, W_test]))