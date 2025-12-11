"""
Schedule-Free Optimizers for Era 2025
=====================================

COSMOS optimizer family combining Muon and SOAP for efficient training.
"""

from .cosmos import COSMOS, Muon, SOAP

__all__ = [
    "COSMOS",
    "Muon", 
    "SOAP"
]

# Optimizer factory
def get_optimizer(optimizer_name: str, model_parameters, **kwargs):
    """Factory function to create optimizer by name."""
    
    if optimizer_name.lower() == 'cosmos':
        return COSMOS(model_parameters, **kwargs)
    elif optimizer_name.lower() == 'muon':
        return Muon(model_parameters, **kwargs)
    elif optimizer_name.lower() == 'soap':
        return SOAP(model_parameters, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

__all__ = ["COSMOSOptimizer", "MuonOptimizer", "SOAPOptimizer"]