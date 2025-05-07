#!/usr/bin/env python3
"""
Convert XGBoost models to ONNX format.
This script provides functionality to convert R XGBoost models to ONNX format.
"""

import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from skl2onnx.common.data_types import FloatTensorType


def extract_xgboost_json(xgb_model):
    """
    Extract the JSON representation of an XGBoost model.
    
    Parameters
    ----------
    xgb_model : object
        XGBoost model object from R
        
    Returns
    -------
    model_json : dict
        JSON representation of the model
    """
    # In R, we would use xgb.dump(model, with_stats=True, dump_format="json")
    # Here, we're assuming the R function has already been called and we have the JSON
    
    if hasattr(xgb_model, 'dump_model'):
        # If the model has a dump_model attribute (XGBoost wrapper)
        model_json = json.loads(xgb_model.dump_model(dump_format='json'))
    else:
        # Try to get model directly if it's already in the right format
        model_json = xgb_model
    
    return model_json


def tree_to_onnx_nodes(tree, tree_id, n_features, n_classes=None):
    """
    Convert a single XGBoost tree to ONNX nodes.
    
    Parameters
    ----------
    tree : dict
        Tree structure from XGBoost JSON model
    tree_id : int
        Tree identifier
    n_features : int
        Number of features
    n_classes : int or None, default=None
        Number of classes for classification
        
    Returns
    -------
    nodes : list
        List of ONNX nodes
    output_name : str
        Name of the output tensor
    """
    # This is a simplified conversion focusing on the core structure
    # A full implementation would handle all XGBoost node types and edge cases
    
    nodes = []
    node_map = {}
    output_map = {}
    
    # Process each node in the tree
    def process_node(node, node_id):
        # Check if node is a leaf
        if 'leaf' in node:
            # Leaf node - return the value
            output_name = f'tree_{tree_id}_node_{node_id}_output'
            leaf_value = node['leaf']
            
            # Create a constant for the leaf value
            value_tensor = numpy_helper.from_array(
                np.array([leaf_value], dtype=np.float32),
                name=f'tree_{tree_id}_node_{node_id}_value'
            )
            
            # Store output name for this node
            output_map[node_id] = output_name
            return value_tensor, output_name
        
        # Otherwise, it's a decision node
        feature_id = node['split']
        threshold = node['split_condition']
        
        # Get left and right children
        left_id = node_id * 2 + 1
        right_id = node_id * 2 + 2
        
        # Process children recursively
        left_tensor, left_output = process_node(node['children'][0], left_id)
        right_tensor, right_output = process_node(node['children'][1], right_id)
        
        # Add tensors to list
        if left_tensor is not None:
            nodes.append(left_tensor)
        if right_tensor is not None:
            nodes.append(right_tensor)
        
        # Create feature access node
        feature_name = f'tree_{tree_id}_node_{node_id}_feature'
        feature_indices = helper.make_tensor(
            name=f'tree_{tree_id}_node_{node_id}_indices',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=np.array([feature_id], dtype=np.int64)
        )
        
        gather_node = helper.make_node(
            'Gather',
            inputs=['input', f'tree_{tree_id}_node_{node_id}_indices'],
            outputs=[feature_name],
            axis=1
        )
        
        # Create comparison node
        threshold_name = f'tree_{tree_id}_node_{node_id}_threshold'
        threshold_tensor = helper.make_tensor(
            name=threshold_name,
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=np.array([threshold], dtype=np.float32)
        )
        
        compare_name = f'tree_{tree_id}_node_{node_id}_compare'
        compare_node = helper.make_node(
            'Less',
            inputs=[feature_name, threshold_name],
            outputs=[compare_name]
        )
        
        # Create conditional selection node
        output_name = f'tree_{tree_id}_node_{node_id}_output'
        if_node = helper.make_node(
            'If',
            inputs=[compare_name, left_output, right_output],
            outputs=[output_name]
        )
        
        # Add all nodes
        nodes.append(feature_indices)
        nodes.append(threshold_tensor)
        nodes.append(gather_node)
        nodes.append(compare_node)
        nodes.append(if_node)
        
        # Store output for this node
        output_map[node_id] = output_name
        
        return None, output_name
    
    # Start tree traversal from root
    _, root_output = process_node(tree, 0)
    
    return nodes, root_output


def xgboost_to_onnx(xgb_model, n_features, n_classes=None, feature_names=None):
    """
    Convert an XGBoost model to ONNX format.
    
    Parameters
    ----------
    xgb_model : object
        XGBoost model object from R
    n_features : int
        Number of features
    n_classes : int or None, default=None
        Number of classes for classification (None for regression)
    feature_names : list or None, default=None
        Names of features (if None, uses X0, X1, etc.)
        
    Returns
    -------
    onnx_model : onnx.ModelProto
        ONNX model
    """
    # Extract JSON model
    model_json = extract_xgboost_json(xgb_model)
    
    # Determine if classification
    is_classification = n_classes is not None and n_classes > 0
    
    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]
    
    # Create ONNX model components
    # Input features
    model_input = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, n_features])
    
    # Convert each tree and extract their outputs
    all_nodes = []
    tree_outputs = []
    
    # Get trees from model (simplified)
    trees = model_json.get('trees', [])
    
    for i, tree in enumerate(trees):
        # Convert tree to ONNX nodes
        nodes, output_name = tree_to_onnx_nodes(
            tree, i, n_features, n_classes)
        
        all_nodes.extend(nodes)
        tree_outputs.append(output_name)
    
    # Aggregate tree outputs
    if is_classification:
        # For multiclass, trees are grouped by class
        if n_classes > 2:
            trees_per_class = len(tree_outputs) // n_classes
            
            # Create outputs for each class
            class_outputs = []
            
            for c in range(n_classes):
                class_tree_outputs = tree_outputs[c * trees_per_class:(c + 1) * trees_per_class]
                
                # Sum trees for this class
                class_sum_name = f'class_{c}_sum'
                
                # Add all tree outputs
                if len(class_tree_outputs) == 1:
                    # Only one tree, no need to add
                    class_sum_name = class_tree_outputs[0]
                else:
                    # Multiple trees, need to add them
                    current_sum = class_tree_outputs[0]
                    
                    for j in range(1, len(class_tree_outputs)):
                        next_sum = f'class_{c}_sum_{j}'
                        add_node = helper.make_node(
                            'Add',
                            inputs=[current_sum, class_tree_outputs[j]],
                            outputs=[next_sum]
                        )
                        all_nodes.append(add_node)
                        current_sum = next_sum
                    
                    class_sum_name = current_sum
                
                class_outputs.append(class_sum_name)
            
            # Combine class outputs
            concat_node = helper.make_node(
                'Concat',
                inputs=class_outputs,
                outputs=['logits'],
                axis=1
            )
            all_nodes.append(concat_node)
            
            # Apply softmax
            softmax_node = helper.make_node(
                'Softmax',
                inputs=['logits'],
                outputs=['probabilities'],
                axis=1
            )
            all_nodes.append(softmax_node)
            
            # Get class prediction
            argmax_node = helper.make_node(
                'ArgMax',
                inputs=['probabilities'],
                outputs=['prediction'],
                axis=1
            )
            all_nodes.append(argmax_node)
            
            # Define model outputs
            model_outputs = [
                helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, n_classes]),
                helper.make_tensor_value_info('prediction', TensorProto.INT64, [None, 1])
            ]
            
        else:
            # Binary classification
            # Sum all tree outputs
            current_sum = tree_outputs[0]
            
            for i in range(1, len(tree_outputs)):
                next_sum = f'tree_sum_{i}'
                add_node = helper.make_node(
                    'Add',
                    inputs=[current_sum, tree_outputs[i]],
                    outputs=[next_sum]
                )
                all_nodes.append(add_node)
                current_sum = next_sum
            
            # Apply sigmoid
            sigmoid_node = helper.make_node(
                'Sigmoid',
                inputs=[current_sum],
                outputs=['probability']
            )
            all_nodes.append(sigmoid_node)
            
            # Create binary probability output
            reshape_node = helper.make_node(
                'Reshape',
                inputs=['probability', 'prob_shape'],
                outputs=['probabilities']
            )
            prob_shape = helper.make_tensor(
                name='prob_shape',
                data_type=TensorProto.INT64,
                dims=[2],
                vals=np.array([-1, 2], dtype=np.int64)
            )
            all_nodes.append(prob_shape)
            all_nodes.append(reshape_node)
            
            # Threshold for class prediction
            threshold_node = helper.make_node(
                'Greater',
                inputs=['probability', 'threshold'],
                outputs=['prediction_bool']
            )
            threshold_tensor = helper.make_tensor(
                name='threshold',
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=np.array([0.5], dtype=np.float32)
            )
            all_nodes.append(threshold_tensor)
            all_nodes.append(threshold_node)
            
            # Convert boolean to integer (0/1)
            cast_node = helper.make_node(
                'Cast',
                inputs=['prediction_bool'],
                outputs=['prediction'],
                to=TensorProto.INT64
            )
            all_nodes.append(cast_node)
            
            # Define model outputs
            model_outputs = [
                helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, 2]),
                helper.make_tensor_value_info('prediction', TensorProto.INT64, [None, 1])
            ]
            
    else:
        # Regression - sum all tree outputs
        current_sum = tree_outputs[0]
        
        for i in range(1, len(tree_outputs)):
            next_sum = f'tree_sum_{i}'
            add_node = helper.make_node(
                'Add',
                inputs=[current_sum, tree_outputs[i]],
                outputs=[next_sum]
            )
            all_nodes.append(add_node)
            current_sum = next_sum
        
        # Add bias if available
        if 'learner_model_param' in model_json and 'base_score' in model_json['learner_model_param']:
            base_score = float(model_json['learner_model_param']['base_score'])
            base_score_tensor = helper.make_tensor(
                name='base_score',
                data_type=TensorProto.FLOAT,
                dims=[1],
                vals=np.array([base_score], dtype=np.float32)
            )
            all_nodes.append(base_score_tensor)
            
            add_bias_node = helper.make_node(
                'Add',
                inputs=[current_sum, 'base_score'],
                outputs=['prediction']
            )
            all_nodes.append(add_bias_node)
        else:
            # No bias, use current sum as prediction
            prediction_node = helper.make_node(
                'Identity',
                inputs=[current_sum],
                outputs=['prediction']
            )
            all_nodes.append(prediction_node)
        
        # Define model outputs
        model_outputs = [
            helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [None, 1])
        ]
    
    # Create graph
    graph = helper.make_graph(
        nodes=all_nodes,
        name='xgboost_model',
        inputs=[model_input],
        outputs=model_outputs
    )
    
    # Create model
    model_name = 'xgboost_classifier' if is_classification else 'xgboost_regressor'
    onnx_model = helper.make_model(
        graph, 
        producer_name='tidylearn',
        opset_imports=[helper.make_opsetid('', 12)]
    )
    
    # Add metadata
    metadata_props = [
        ('model_type', model_name),
        ('n_trees', str(len(trees))),
        ('n_features', str(n_features)),
        ('feature_names', ','.join(feature_names))
    ]
    
    if is_classification:
        metadata_props.append(('n_classes', str(n_classes)))
    
    for key, value in metadata_props:
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = value
    
    # Note: This implementation is a simplified version focusing on the core conversion.
    # A complete implementation would need to handle:
    # 1. Full XGBoost tree structure with all node types
    # 2. XGBoost-specific parameters and configurations
    # 3. Proper handling of multiclass models
    # 4. Optimization of the ONNX graph
    
    return onnx_model


if __name__ == "__main__":
    # Example usage (using a mock XGBoost model)
    # In practice, this model would come from R's XGBoost library
    mock_xgb_model = {
        "learner_model_param": {"base_score": "0.5"},
        "trees": [
            {
                "nodeid": 0,
                "split": 2,
                "split_condition": 0.5,
                "yes": 1,
                "no": 2,
                "children": [
                    {"nodeid": 1, "leaf": 1.5},
                    {"nodeid": 2, "leaf": -0.5}
                ]
            }
        ]
    }
    
    # Convert to ONNX
    model = xgboost_to_onnx(
        mock_xgb_model,
        n_features=5,
        feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
    )
    
    # Save model to file
    onnx.save(model, "xgboost_model.onnx")
