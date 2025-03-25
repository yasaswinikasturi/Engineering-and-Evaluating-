class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'

    # Model Settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Chained Multi-outputs Settings
    CHAINED_LEVELS = {
        'level1': ['y2'],
        'level2': ['y2', 'y3'],
        'level3': ['y2', 'y3', 'y4']
    }
    
    # Hierarchical Settings
    HIERARCHICAL_LEVELS = {
        'level1': ['y2'],
        'level2': ['y3'],
        'level3': ['y4']
    }

    # Model Parameters
    MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
