import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from Config import Config
from collections import defaultdict
from sklearn.metrics import classification_report

class ChainedModel(BaseModel):
    """
    Strategy implementation for Chained Multi-outputs classification model.
    Implements the Strategy pattern by providing concrete implementation of BaseModel methods.
    """
    def __init__(self):
        super().__init__()
        self.models = {}
        self.levels = Config.CHAINED_LEVELS
        self.label_encoders = {}
        self.model_name = "Chained Multi-outputs Model"

    def train(self, X, y):
        """
        Train models for each chained level
        
        Args:
            X: Input features
            y: Target labels DataFrame containing all label columns
        """
        for level, labels in self.levels.items():
            print(f"Training {level} model with labels: {labels}")
            
            # Create combined labels for this level
            if len(labels) == 1:
                # Handle single label case
                y_target = y[labels[0]].values
                if len(y_target.shape) > 1 and y_target.shape[1] == 1:
                    y_target = y_target.ravel()  # Convert column vector to 1D array
            else:
                # Handle multi-label case
                y_combined = np.column_stack([y[label] for label in labels])
                y_target = y_combined
            
            # Train model for this level
            model = RandomForestClassifier(**Config.MODEL_PARAMS)
            model.fit(X, y_target)
            self.models[level] = model
            self.is_trained = True

    def predict(self, X):
        """
        Make predictions for each chained level
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions for each level
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = {}
        for level, labels in self.levels.items():
            # Get predictions for this level
            pred = self.models[level].predict(X)
            predictions[level] = pred

        return predictions

    def print_results(self, y_true, y_pred):
        """
        Print chained model results in tabular format
        
        Args:
            y_true: True labels DataFrame
            y_pred: Dictionary with predictions for each level
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before printing results")

        print(f"\n{'-'*20} {self.model_name} Results {'-'*20}")
        
        overall_metrics = defaultdict(list)
        
        for i, (level, labels) in enumerate(self.levels.items()):
            print(f"\n[Level {i+1}]: {level} - Labels: {labels}")
            
            # Handle single label case differently
            if len(labels) == 1:
                # For single label, use direct accuracy comparison
                label = labels[0]
                pred = y_pred[level].flatten() if y_pred[level].ndim > 1 else y_pred[level]
                true = y_true[label].values
                
                # Get detailed metrics
                metrics = self._get_metrics(true, pred)
                
                # Format as table
                headers = list(metrics.keys())
                data = [list(metrics.values())]
                print(self._format_table(data, headers))
                
                # Add to overall metrics
                for k, v in metrics.items():
                    overall_metrics[k].append(v)
                
                # Print classification report
                print(f"\nDetailed Classification Report for {label}:")
                print(self._get_classification_report(true, pred))
                
            else:
                # For multiple labels, calculate metrics for each label
                y_true_combined = np.column_stack([y_true[label] for label in labels])
                pred_combined = y_pred[level]
                
                # Calculate overall hierarchical accuracy
                hier_acc = self._calculate_hierarchical_accuracy(y_true_combined, pred_combined)
                print(f"Hierarchical Accuracy: {hier_acc:.4f}")
                
                # For each individual label in the combination
                print("\nIndividual Label Metrics:")
                all_metrics = []
                
                for i, label in enumerate(labels):
                    true_label = y_true[label].values
                    pred_label = pred_combined[:, i] if pred_combined.ndim > 1 else pred_combined
                    
                    metrics = self._get_metrics(true_label, pred_label)
                    metrics_with_label = {'Label': label, **metrics}
                    all_metrics.append(list(metrics_with_label.values()))
                    
                    # Add to overall metrics
                    for k, v in metrics.items():
                        overall_metrics[k].append(v)
                
                # Print as table
                headers = ['Label', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
                print(self._format_table(all_metrics, headers))
                
        # Print overall metrics
        print(f"\n{'-'*20} Overall Model Performance {'-'*20}")
        avg_metrics = {k: sum(v)/len(v) for k, v in overall_metrics.items()}
        headers = list(avg_metrics.keys())
        data = [list(avg_metrics.values())]
        print(self._format_table(data, headers)) 