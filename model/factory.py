from .chained_model import ChainedModel
from .hierarchical_model import HierarchicalModel
from .base_model import BaseModel
from enum import Enum

class ModelType(Enum):
    CHAINED = "chained"
    HIERARCHICAL = "hierarchical"

class ModelFactory:
    """
    Factory class for creating model instances based on the specified model type.
    Implements the Factory Pattern to create different model strategies.
    """
    
    @staticmethod
    def create_model(model_type: ModelType) -> BaseModel:
        """
        Create and return a model instance based on the specified model type.
        
        Args:
            model_type: Type of model to create (from ModelType enum)
            
        Returns:
            Instance of the requested model type
        """
        if model_type == ModelType.CHAINED:
            return ChainedModel()
        elif model_type == ModelType.HIERARCHICAL:
            return HierarchicalModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 