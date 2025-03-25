from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from Config import Config
from tabulate import tabulate

class BaseModel(ABC):
    """
    Abstract base class defining the strategy interface for all classification models.
    Implements the Strategy Pattern where concrete strategies implement the model behavior.
    """
    def __init__(self):
        self.model = RandomForestClassifier(**Config.MODEL_PARAMS)
        self.is_trained = False
        self.model_name = self.__class__.__name__

    @abstractmethod
    def train(self, X, y):
        """
        Train the model strategy
        
        Args:
            X: Input features
            y: Target labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the model strategy
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass

    @abstractmethod
    def print_results(self, y_true, y_pred):
        """
        Print model results in a tabular format
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        pass

    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy for predictions"""
        return np.mean(y_true == y_pred)

    def _calculate_hierarchical_accuracy(self, y_true, y_pred):
        """Calculate hierarchical accuracy for multi-label predictions"""
        return np.mean(np.all(y_true == y_pred, axis=1))
        
    def _get_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive metrics for model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return metrics
    
    def _get_classification_report(self, y_true, y_pred):
        """
        Get classification report with proper handling of zero division cases
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, zero_division=0)
        
    def _format_table(self, data, headers):
        """
        Format data as a table using tabulate
        
        Args:
            data: Table data
            headers: Table headers
            
        Returns:
            Formatted table string
        """
        return tabulate(data, headers=headers, tablefmt="grid") 