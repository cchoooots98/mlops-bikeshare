# src/inference/handler.py
# AWS Lambda handler for bikeshare predictor
# Purpose: Entry point for Lambda function that runs batch inference

import json
import traceback
from src.inference.predictor import main


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event object (can be empty {} or contain configuration)
        context: Lambda context object
    
    Returns:
        dict: Response with status and message
    """
    try:
        # Run the main predictor logic
        main()
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "message": "Predictor executed successfully"
            })
        }
    except Exception as e:
        # Log the error and return error response
        error_msg = f"Predictor failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "message": error_msg
            })
        }
