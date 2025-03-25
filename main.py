from preprocess import *
from embeddings import *
from model.factory import ModelFactory, ModelType
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from Config import Config

# Set random seeds
seed = Config.RANDOM_STATE
random.seed(seed)
np.random.seed(seed)

def load_data():
    """Load input data from CSV files"""
    print("Loading data...")
    df = get_input_data()
    return df

def preprocess_data(df):
    """Preprocess data with noise removal and deduplication"""
    print("Preprocessing data...")
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    return df

def get_embeddings(df: pd.DataFrame):
    """Convert text data to numerical embeddings"""
    print("Generating embeddings...")
    X = get_tfidf_embd(df)  # Get tf-idf embeddings
    return X, df

def prepare_labels(df: pd.DataFrame):
    """Prepare labels for multi-label classification"""
    print("Preparing labels...")
    labels = {}
    for col in Config.TYPE_COLS:
        if col in df.columns:
            # Convert to numpy array and handle missing values
            labels[col] = df[col].fillna('').values
    return pd.DataFrame(labels)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    model.train(X_train, y_train)
    
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    model.print_results(y_test, y_pred)
    
    return model

def main():
    try:
        # Load and preprocess data
        df = load_data()
        df = preprocess_data(df)
        
        # Convert text columns to string type
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
        
        # Group data by Type 1
        grouped_df = df.groupby(Config.GROUPED)
        
        for name, group_df in grouped_df:
            print(f"\n{'-'*30} Processing group: {name} {'-'*30}")
            
            # Get embeddings
            X, group_df = get_embeddings(group_df)
            
            # Prepare labels
            y = prepare_labels(group_df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            
            print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
            
            # Use factory to create models - this is the key Factory Pattern implementation
            print("\nCreating models using Factory Pattern...")
            
            # Create and train Chained Model (Strategy 1)
            chained_model = ModelFactory.create_model(ModelType.CHAINED)
            train_and_evaluate_model(chained_model, X_train, X_test, y_train, y_test, "Chained Model")
            
            # Create and train Hierarchical Model (Strategy 2)
            hierarchical_model = ModelFactory.create_model(ModelType.HIERARCHICAL)
            train_and_evaluate_model(hierarchical_model, X_train, X_test, y_train, y_test, "Hierarchical Model")
            
            print(f"\n{'-'*30} Completed processing group: {name} {'-'*30}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

