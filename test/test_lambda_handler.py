# Test for AWS Lambda handler
# Basic tests to verify the handler module structure


def test_handler_importable():
    """Verify the Lambda handler can be imported."""
    from src.inference.handler import lambda_handler
    
    assert callable(lambda_handler)


def test_handler_signature():
    """Verify the Lambda handler has the correct signature."""
    from src.inference.handler import lambda_handler
    import inspect
    
    sig = inspect.signature(lambda_handler)
    params = list(sig.parameters.keys())
    
    # Lambda handlers must accept 'event' and 'context' parameters
    assert 'event' in params
    assert 'context' in params
