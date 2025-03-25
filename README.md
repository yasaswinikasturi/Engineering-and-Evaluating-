# Multi-Label Email Classification System

## Overview
This project implements a multi-label classification system for email categorization using two distinct architectural approaches. It's designed to classify emails into multiple hierarchical categories (Type 1, Type 2, Type 3, and Type 4), demonstrating software design patterns and principles for building modular, maintainable AI systems.

## Key Features
- **Modular Architecture**: Clear separation between preprocessing, embedding generation, and modeling
- **Design Patterns**: Factory Pattern and Strategy Pattern implementation
- **Multiple Classification Strategies**:
  - Chained Multi-output Classification
  - Hierarchical Modeling Classification
- **Evaluation Metrics**: Comprehensive performance metrics for model comparison

## System Requirements
- Python 3.8 or higher
- Required packages (listed in requirements.txt):
  - numpy >= 1.19.0
  - pandas >= 1.0.0
  - scikit-learn >= 0.24.0
  - tabulate >= 0.8.7
  - stanza >= 1.3.0
  - transformers >= 4.12.0
  - matplotlib >= 3.3.0
  - seaborn >= 0.11.0

## Installation

1. Clone the repository or extract the project files
2. Navigate to the project directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## File Structure

```
├── Config.py                  # Configuration parameters
├── data/                      # Input data directory
│   ├── AppGallery.csv         # AppGallery & Games data
│   └── Purchasing.csv         # In-App Purchase data
├── embeddings.py              # Text-to-numeric conversion
├── main.py                    # Main controller
├── model/                     # Model implementations
│   ├── base_model.py          # Abstract base class
│   ├── chained_model.py       # Chained multi-output strategy
│   ├── factory.py             # Model factory implementation
│   └── hierarchical_model.py  # Hierarchical modeling strategy
├── preprocess.py              # Data preprocessing functions
└── requirements.txt           # Package dependencies
```

## Architecture Design

### 1. Separation of Concerns
The system implements separation of concerns with:
- Data loading and preprocessing (`preprocess.py`)
- Text-to-numeric conversion (`embeddings.py`)
- Model training and evaluation (`model/` directory)
- Configuration parameters (`Config.py`)
- Main controller (`main.py`)

### 2. Input Data Consistency
- Consistent data formats using pandas DataFrames 
- Standardized embedding generation (TF-IDF)
- Shared configuration across all components

### 3. Multi-Label Classification Approaches

#### Chained Multi-outputs (Design Decision 1)
This approach uses single model instances to predict multiple levels in a chained manner:
1. Level 1: Predicts Type 2 (y2)
2. Level 2: Predicts Type 2 + Type 3 (y2, y3)
3. Level 3: Predicts Type 2 + Type 3 + Type 4 (y2, y3, y4)

The model creates a unified prediction for all levels, accounting for dependencies between types.

#### Hierarchical Modeling (Design Decision 2)
This approach creates a tree of model instances based on preceding class values:
1. Root level: Predicts Type 2 (y2)
2. For each unique value in Type 2: Train a model to predict Type 3
3. For each unique value combination of Type 2 and Type 3: Train a model to predict Type 4

Data is filtered at each level, with separate models specialized for each class combination.

### 4. Design Patterns

#### Factory Pattern
The `ModelFactory` class creates model instances based on the specified type:
```python
@staticmethod
def create_model(model_type: ModelType) -> BaseModel:
    if model_type == ModelType.CHAINED:
        return ChainedModel()
    elif model_type == ModelType.HIERARCHICAL:
        return HierarchicalModel()
```

#### Strategy Pattern
The `BaseModel` abstract class defines a standard interface that all models must implement:
```python
@abstractmethod
def train(self, X, y):
    pass

@abstractmethod
def predict(self, X):
    pass

@abstractmethod
def print_results(self, y_true, y_pred):
    pass
```

## How to Run

Simply execute the main.py script:

```bash
python main.py
```

The system will:
1. Load and preprocess data from CSV files
2. Generate embeddings using TF-IDF
3. Train and evaluate both model types for each data group
4. Display comprehensive performance metrics

## Results

The system processes two data groups and evaluates both model approaches:

### AppGallery & Games Group

#### Chained Multi-outputs Model
- **Level 1 (Type 2)**: 76% accuracy, 0.78 precision, 0.76 recall, 0.73 F1-score
- **Level 2 (Type 2+3)**: 56% hierarchical accuracy
- **Level 3 (Type 2+3+4)**: 52% hierarchical accuracy
- **Overall**: 72.67% average accuracy across all levels

#### Hierarchical Model
- **First Level (Type 2)**: 76% accuracy, 0.78 precision, 0.76 recall, 0.73 F1-score
- **Second Level**: Accuracy varies by Type 2 value (40-100%)
- **Third Level**: Accuracy varies by Type 2 + Type 3 combination (8-100%)
- **Overall**: 72.64% average accuracy across all levels

### In-App Purchase Group

#### Chained Multi-outputs Model
- **Level 1 (Type 2)**: 82.35% accuracy, 0.77 precision, 0.82 recall, 0.80 F1-score
- **Level 2 (Type 2+3)**: 70.59% hierarchical accuracy
- **Level 3 (Type 2+3+4)**: 52.94% hierarchical accuracy
- **Overall**: 73.53% average accuracy across all levels

#### Hierarchical Model
- **First Level (Type 2)**: 82.35% accuracy, 0.77 precision, 0.82 recall, 0.80 F1-score
- **Second Level**: Accuracy varies by Type 2 value (50-67%)
- **Third Level**: Accuracy varies by Type 2 + Type 3 combination (50-53%)
- **Overall**: 60.47% average accuracy across all levels

## Performance Comparison

1. **Overall Accuracy**:
   - Chained approach generally shows slightly better overall performance
   - Hierarchical approach provides more detailed insights for specific combinations

2. **Model Complexity**:
   - Chained approach uses fewer models (3 levels per group)
   - Hierarchical approach creates more models (one per class combination)

3. **Prediction Insights**:
   - Chained approach better captures dependencies between types
   - Hierarchical approach offers insights into performance for specific class combinations

## Conclusion

Both architectural approaches successfully implement multi-label classification with different trade-offs:

- **Chained Multi-outputs** provides better overall performance and a simpler model structure
- **Hierarchical Modeling** offers more granular insights into class-specific performance

The implementation demonstrates effective use of software architecture principles and design patterns for building modular, maintainable AI systems. 