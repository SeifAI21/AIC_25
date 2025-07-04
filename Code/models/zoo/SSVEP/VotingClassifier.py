"""
SSVEP Voting Classifier Model
Implements ensemble of classical ML models for SSVEP classification
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgb
import catboost as cb
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SSVEPVotingClassifier:
    """
    Voting classifier ensemble for SSVEP classification
    Uses the top 3 classifiers identified in the notebook: LDA, LightGBM, CatBoost
    """
    
    def __init__(self, num_classes, voting='soft', random_state=42):
        """
        Initialize the voting classifier
        
        Args:
            num_classes: Number of SSVEP classes
            voting: 'soft' or 'hard' voting
            random_state: Random seed for reproducibility
        """
        self.num_classes = num_classes
        self.voting = voting
        self.random_state = random_state
        self.is_fitted = False
        
        # Define base classifiers
        self.base_classifiers = {
            'lda': LinearDiscriminantAnalysis(),
            'lgbm': lgb.LGBMClassifier(random_state=random_state, verbose=-1),
            'cat': cb.CatBoostClassifier(verbose=False, random_state=random_state)
        }
        
        # Create ensemble with scaling pipelines (matching notebook approach)
        estimators = []
        for name in ['cat', 'lda', 'lgbm']:  # Match notebook order
            clf = clone(self.base_classifiers[name])
            pipe = make_pipeline(StandardScaler(), clf)
            estimators.append((name, pipe))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        """
        Fit the voting classifier
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self.ensemble.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.voting == 'soft':
            return self.ensemble.predict_proba(X)
        else:
            # For hard voting, return one-hot encoded predictions
            predictions = self.predict(X)
            probas = np.zeros((len(predictions), self.num_classes))
            probas[np.arange(len(predictions)), predictions] = 1.0
            return probas
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Mean accuracy
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        return self.ensemble.score(X, y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'num_classes': self.num_classes,
            'voting': self.voting,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self 