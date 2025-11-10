#!/usr/bin/env python3
"""
Convert linear/logistic regression models to ONNX format.
This script provides functionality to convert R linear models to ONNX format.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
from skl2onnx.common.data_types import FloatTensorType


def linear_model_to_onnx(coefficients, is_classification=False, 
                         feature_names=None, target_names=None):
    """
    Convert a linear/logistic regression model to ONNX format.
    
    Parameters
    ----------
    coefficients : dict or array-like
        Model coefficients including intercept
    is_classification : bool, default=False
        Whether this is a classification model
    feature_names : list or None, default=None
        Names of features (if None, uses X0, X1, etc.)
    target_names : list or None, default=None
        Names of target classes for classification (if None, uses Y0, Y1, etc.)
        
    Returns
    -------
    onnx_model : onnx.ModelProto
        ONNX model
    """
    # Convert coefficients to numpy array
    if isinstance(coefficients, dict):
        coef_array = np.array(list(coefficients.values()))
    else:
        coef_array = np.array(coefficients)
    
    # Extract intercept and coefficients
    intercept = coef_array[0]
    coefficients = coef_array[1:]
    
    # Default feature names if not provided
    n_features = len(coefficients)
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]
    
    # Create ONNX model components
    # Input features
    model_input = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, n_features])
    
    # Create nodes
    if is_classification:
        # For logistic regression
        if target_names is None:
            # Default target names for binary classification
            target_names = ["Y0", "Y1"]
        
        # Coefficients tensor (including intercept)
        coef_tensor = helper.make_tensor(
            name='coefficients',
            data_type=TensorProto.FLOAT,
            dims=[1, n_features],
            vals=coefficients.astype(np.float32)
        )
        
        intercept_tensor = helper.make_tensor(
            name='intercept',
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=np.array([intercept]).astype(np.float32)
        )
        
        # Linear combination node
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'coefficients'],
            outputs=['linear_combination']
        )
        
        # Add intercept
        add_node = helper.make_node(
            'Add',
            inputs=['linear_combination', 'intercept'],
            outputs=['logits']
        )
        
        # Apply sigmoid for binary classification
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=['logits'],
            outputs=['probabilities']
        )
        
        # Threshold for class prediction
        threshold_tensor = helper.make_tensor(
            name='threshold',
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=np.array([0.5]).astype(np.float32)
        )
        
        # Apply threshold to get class
        greater_node = helper.make_node(
            'Greater',
            inputs=['probabilities', 'threshold'],
            outputs=['prediction_bool']
        )
        
        # Convert boolean to integer (0/1)
        cast_node = helper.make_node(
            'Cast',
            inputs=['prediction_bool'],
            outputs=['prediction'],
            to=TensorProto.INT64
        )
        
        # Define model outputs
        model_outputs = [
            helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info('prediction', TensorProto.INT64, [None, 1])
        ]
        
        # Create graph
        graph = helper.make_graph(
            nodes=[matmul_node, add_node, sigmoid_node, greater_node, cast_node],
            name='logistic_regression',
            inputs=[model_input],
            outputs=model_outputs,
            initializer=[coef_tensor, intercept_tensor, threshold_tensor]
        )
        
    else:
        # For linear regression
        # Coefficients tensor (including intercept)
        coef_tensor = helper.make_tensor(
            name='coefficients',
            data_type=TensorProto.FLOAT,
            dims=[1, n_features],
            vals=coefficients.astype(np.float32)
        )
        
        intercept_tensor = helper.make_tensor(
            name='intercept',
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=np.array([intercept]).astype(np.float32)
        )
        
        # Linear combination node
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'coefficients'],
            outputs=['linear_combination']
        )
        
        # Add intercept
        add_node = helper.make_node(
            'Add',
            inputs=['linear_combination', 'intercept'],
            outputs=['prediction']
        )
        
        # Define model outputs
        model_outputs = [
            helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [None, 1])
        ]
        
        # Create graph
        graph = helper.make_graph(
            nodes=[matmul_node, add_node],
            name='linear_regression',
            inputs=[model_input],
            outputs=model_outputs,
            initializer=[coef_tensor, intercept_tensor]
        )
    
    # Create model
    model_name = 'logistic_regression' if is_classification else 'linear_regression'
    onnx_model = helper.make_model(
        graph, 
        producer_name='tidysl',
        opset_imports=[helper.make_opsetid('', 12)]
    )
    
    # Add metadata
    metadata_props = [
        ('model_type', model_name),
        ('n_features', str(n_features)),
        ('feature_names', ','.join(feature_names))
    ]
    
    if is_classification:
        metadata_props.append(('target_names', ','.join(target_names)))
    
    for key, value in metadata_props:
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = value
    
    # Check model validity
    onnx.checker.check_model(onnx_model)
    
    return onnx_model


if __name__ == "__main__":
    # Example usage
    coefs = {'intercept': 1.5, 'X1': 0.3, 'X2': -0.5, 'X3': 0.8}
    model = linear_model_to_onnx(coefs, is_classification=False, 
                               feature_names=['X1', 'X2', 'X3'])
    
    # Save model to file
    onnx.save(model, "linear_model.onnx")
