# ---------------------------------------------
# Import general utility functions
# ---------------------------------------------


from models.data_config import FEATURE_ALLOWED_FEATURES

def validate_feature_id(feature_id: str) -> bool:
    """
    Validate the given feature ID
        
    Args:
        feature_id (str)
    
    Returns:
        is_ok (bool)
    """
    if feature_id not in FEATURE_ALLOWED_FEATURES:
        return False
    
    return True