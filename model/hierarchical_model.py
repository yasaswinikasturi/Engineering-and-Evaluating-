import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel
from Config import Config
from collections import defaultdict
from sklearn.metrics import classification_report

class HierarchicalModel(BaseModel):
    """
    Strategy implementation for Hierarchical classification model.
    Implements the Strategy pattern by providing concrete implementation of BaseModel methods.
    """
    def __init__(self):
        super().__init__()
        self.models = {}
        self.levels = Config.HIERARCHICAL_LEVELS
        self.label_encoders = {}
        self.model_name = "Hierarchical Model"
        self.model_stats = defaultdict(dict)  # Track stats for each model

    def _create_model_key(self, parent_labels):
        """Create a unique key for model instances"""
        return '_'.join([f"{k}={v}" for k, v in parent_labels.items()])

    def train(self, X, y):
        """
        Train hierarchical models with filtering approach
        
        Args:
            X: Input features
            y: Target labels DataFrame containing all label columns
        """
        # Train first level model
        first_level = self.levels['level1'][0]
        print(f"Training root model for {first_level}")
        self.models['root'] = RandomForestClassifier(**Config.MODEL_PARAMS)
        self.models['root'].fit(X, y[first_level])
        self.model_stats['root'] = {'data_size': len(X)}
        
        # Train second level models
        unique_first_level = np.unique(y[first_level])
        for label in unique_first_level:
            # Filter data for this first level label
            mask = y[first_level] == label
            if np.sum(mask) > 0:
                X_filtered = X[mask]
                y_filtered = y[mask]
                
                # Train model for second level
                model_key = self._create_model_key({first_level: label})
                print(f"Training second level model: {model_key} with {np.sum(mask)} samples")
                self.models[model_key] = RandomForestClassifier(**Config.MODEL_PARAMS)
                self.models[model_key].fit(X_filtered, y_filtered[self.levels['level2'][0]])
                self.model_stats[model_key] = {'data_size': len(X_filtered)}
                
                # Train third level models
                unique_second_level = np.unique(y_filtered[self.levels['level2'][0]])
                for label2 in unique_second_level:
                    mask2 = y_filtered[self.levels['level2'][0]] == label2
                    if np.sum(mask2) > 0:
                        X_filtered2 = X_filtered[mask2]
                        y_filtered2 = y_filtered[mask2]
                        
                        # Train model for third level
                        model_key2 = self._create_model_key({
                            first_level: label,
                            self.levels['level2'][0]: label2
                        })
                        print(f"Training third level model: {model_key2} with {np.sum(mask2)} samples")
                        self.models[model_key2] = RandomForestClassifier(**Config.MODEL_PARAMS)
                        self.models[model_key2].fit(X_filtered2, y_filtered2[self.levels['level3'][0]])
                        self.model_stats[model_key2] = {'data_size': len(X_filtered2)}
        
        self.is_trained = True
        
        # Print model hierarchy summary
        print("\nHierarchical Model Structure:")
        model_data = []
        for key, stats in self.model_stats.items():
            model_data.append([key, stats['data_size']])
        print(self._format_table(model_data, ['Model Key', 'Training Data Size']))

    def predict(self, X):
        """
        Make hierarchical predictions using the filtering approach
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions for each level and combination
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = {}
        
        # First level prediction
        first_level = self.levels['level1'][0]
        pred_first = self.models['root'].predict(X)
        predictions[first_level] = pred_first
        
        # Second level predictions
        unique_first_level = np.unique(pred_first)
        for label in unique_first_level:
            mask = pred_first == label
            if np.sum(mask) > 0:
                X_filtered = X[mask]
                model_key = self._create_model_key({first_level: label})
                if model_key in self.models:
                    pred_second = self.models[model_key].predict(X_filtered)
                    predictions[f"{first_level}_{label}"] = pred_second
                    
                    # Third level predictions
                    unique_second_level = np.unique(pred_second)
                    for label2 in unique_second_level:
                        mask2 = pred_second == label2
                        if np.sum(mask2) > 0:
                            X_filtered2 = X_filtered[mask2]
                            model_key2 = self._create_model_key({
                                first_level: label,
                                self.levels['level2'][0]: label2
                            })
                            if model_key2 in self.models:
                                pred_third = self.models[model_key2].predict(X_filtered2)
                                predictions[f"{first_level}_{label}_{label2}"] = pred_third

        return predictions

    def print_results(self, y_true, y_pred):
        """
        Print hierarchical model results in tabular format
        
        Args:
            y_true: True labels DataFrame
            y_pred: Dictionary with predictions for each level and combination
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before printing results")

        print(f"\n{'-'*20} {self.model_name} Results {'-'*20}")
        
        overall_metrics = defaultdict(list)
        
        # First level results
        first_level = self.levels['level1'][0]
        metrics_first = self._get_metrics(y_true[first_level], y_pred[first_level])
        
        print(f"\nFirst Level ({first_level}):")
        headers = list(metrics_first.keys())
        data = [list(metrics_first.values())]
        print(self._format_table(data, headers))
        
        # Add to overall metrics
        for k, v in metrics_first.items():
            overall_metrics[k].append(v)
        
        # Classification report for first level
        print(f"\nDetailed Classification Report for {first_level}:")
        print(self._get_classification_report(y_true[first_level], y_pred[first_level]))
        
        # Second level results
        print(f"\n{'-'*20} Second Level Results {'-'*20}")
        unique_first_level = np.unique(y_pred[first_level])
        
        # Prepare data for table
        second_level_metrics = []
        
        for label in unique_first_level:
            mask = y_pred[first_level] == label
            if np.sum(mask) > 0:
                model_key = self._create_model_key({first_level: label})
                if model_key in self.models:
                    y_true_filtered = y_true[mask]
                    y_pred_filtered = y_pred[f"{first_level}_{label}"]
                    metrics = self._get_metrics(
                        y_true_filtered[self.levels['level2'][0]],
                        y_pred_filtered
                    )
                    
                    # Add parent label info to metrics
                    row_data = [f"{first_level}={label}", np.sum(mask)]
                    row_data.extend(list(metrics.values()))
                    second_level_metrics.append(row_data)
                    
                    # Add to overall metrics
                    for k, v in metrics.items():
                        overall_metrics[k].append(v)
        
        # Print table for second level metrics
        if second_level_metrics:
            headers = ['Parent Label', 'Samples', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            print(self._format_table(second_level_metrics, headers))
        
        # Third level results
        print(f"\n{'-'*20} Third Level Results {'-'*20}")
        third_level_metrics = []
        
        for label in unique_first_level:
            mask = y_pred[first_level] == label
            if np.sum(mask) > 0:
                y_pred_filtered = y_pred.get(f"{first_level}_{label}", None)
                if y_pred_filtered is not None:
                    unique_second_level = np.unique(y_pred_filtered)
                    for label2 in unique_second_level:
                        mask2 = y_pred_filtered == label2
                        if np.sum(mask2) > 0:
                            model_key2 = self._create_model_key({
                                first_level: label,
                                self.levels['level2'][0]: label2
                            })
                            if model_key2 in self.models:
                                filtered_indices = np.where(mask)[0][mask2]
                                y_true_filtered2 = y_true.iloc[filtered_indices]
                                y_pred_filtered2 = y_pred.get(f"{first_level}_{label}_{label2}", None)
                                if y_pred_filtered2 is not None:
                                    metrics = self._get_metrics(
                                        y_true_filtered2[self.levels['level3'][0]],
                                        y_pred_filtered2
                                    )
                                    
                                    # Add parent labels info to metrics
                                    row_data = [f"{first_level}={label}, {self.levels['level2'][0]}={label2}", np.sum(mask2)]
                                    row_data.extend(list(metrics.values()))
                                    third_level_metrics.append(row_data)
                                    
                                    # Add to overall metrics
                                    for k, v in metrics.items():
                                        overall_metrics[k].append(v)
        
        # Print table for third level metrics
        if third_level_metrics:
            headers = ['Parent Labels', 'Samples', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            print(self._format_table(third_level_metrics, headers))
        
        # Print overall model performance
        print(f"\n{'-'*20} Overall Model Performance {'-'*20}")
        avg_metrics = {k: sum(v)/len(v) for k, v in overall_metrics.items()}
        headers = list(avg_metrics.keys())
        data = [list(avg_metrics.values())]
        print(self._format_table(data, headers)) 