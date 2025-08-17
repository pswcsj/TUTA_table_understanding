#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed evaluation with precision, recall, F1 scores
"""

import torch
import json
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
import tokenizer as tknr
import multilabel_minimal
from dataset_types import LABEL_MAPS
from collections import defaultdict
import sys
import time

def load_model():
    """Load the trained model"""
    args = argparse.Namespace(
        vocab_path='./vocab/bert_vocab.txt',
        context_repo_path='./vocab/context_repo_init.txt',
        cellstr_repo_path='./vocab/cellstr_repo_init.txt',
        hidden_size=768,
        intermediate_size=3072,
        magnitude_size=10,
        precision_size=10,
        top_digit_size=10,
        low_digit_size=10,
        row_size=256,
        column_size=256,
        tree_depth=4,
        node_degree=[32, 32, 64, 256],
        num_format_feature=11,
        attention_distance=8,
        attention_step=0,
        num_attention_heads=12,
        num_encoder_layers=12,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        layer_norm_eps=1e-6,
        hidden_act='gelu',
        target='multilabel',
        attn_method='add',
        aggregator='avg',
        cell_pooling='mean',
        max_seq_len=512,
        max_cell_num=256,
        max_cell_length=64,
        add_separate=True,
        text_threshold=0.5,
        value_threshold=0.1,
        max_disturb_num=20,
        disturb_prob=0.15,
        wcm_rate=0.3,
        clc_rate=0.3,
        hier_or_flat='both',
        batch_size=1
    )
    
    print("Loading model...")
    args.tokenizer = tknr.TutaTokenizer(args)
    args.vocab_size = len(args.tokenizer.vocab)
    
    model = multilabel_minimal.TUTAMultiLabelNative(args)
    checkpoint = torch.load('./models/model-final', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✓ Model loaded ({len(checkpoint)} parameters)")
    return model, args

def evaluate_table_detailed(model, table_data, args):
    """Evaluate a single table and return predictions and ground truth"""
    
    reader = multilabel_minimal.MultiLabelTableReader(args)
    tokenizer = multilabel_minimal.MultiLabelTableTokenizer(args)
    
    # Convert to hierarchical format if needed
    if "cells" in table_data:
        table_data = multilabel_minimal.convert_to_hierarchical(table_data)
    
    result = reader.get_inputs(table_data)
    if result is None:
        return None, None
    
    string_matrix, position_lists, header_info, format_matrix = result
    top_position_list, left_position_list = position_lists
    
    # Get ground truth labels from cells
    label_matrices = {}
    for task in LABEL_MAPS.keys():
        label_matrices[task] = reader.get_label_matrix(table_data["Cells"], task)
    
    tokenized = tokenizer.tokenize_multilabel(
        string_matrix,
        top_position_list,
        left_position_list,
        format_matrix,
        label_matrices,
        header_info,
    )
    
    if tokenized is None:
        return None, None
    
    batch = multilabel_minimal.collate_multilabel([tokenized])
    
    # Get predictions
    predictions = {}
    ground_truth = {}
    
    with torch.no_grad():
        # Unpack batch
        if len(batch) == 9:  # TUTA format
            (token_id, num_matrix, pos_row, pos_col, pos_top, pos_left, 
             format_vec, indicator, batch_labels) = batch
        else:
            (token_id, num_matrix, pos_top, pos_left, 
             format_vec, indicator, batch_labels) = batch
            pos_row = pos_col = None
        
        # Extract numerical features
        num_mag = num_matrix[:, :, 0]
        num_pre = num_matrix[:, :, 1]
        num_top = num_matrix[:, :, 2]
        num_low = num_matrix[:, :, 3]
        token_order = torch.zeros_like(token_id)
        
        # Get encoded states
        if pos_row is not None:
            encoded_states = model.backbone(
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_row, pos_col, pos_top, pos_left,
                format_vec, indicator
            )
        else:
            encoded_states = model.backbone(
                token_id, num_mag, num_pre, num_top, num_low,
                token_order, pos_top, pos_left,
                format_vec, indicator
            )
        
        # Aggregate to cell level
        aggregated = model.multi_label_head.aggr_funcs[args.aggregator](
            encoded_states, indicator
        )
        
        # Get predictions for each task
        for task in LABEL_MAPS.keys():
            pred_list = []
            true_list = []
            
            # Get valid cells
            valid_cells = (indicator[0] > 0).nonzero(as_tuple=True)[0]
            
            for cell_idx in valid_cells:
                cell_id = indicator[0][cell_idx].item()
                
                # Get cell feature
                cell_feature = aggregated[0, cell_idx].unsqueeze(0)
                cell_feature = model.multi_label_head.uniform_linear(cell_feature)
                cell_feature = model.multi_label_head.tanh(cell_feature)
                
                # Get prediction
                logits = model.multi_label_head.classifiers[task](cell_feature)
                pred = logits.argmax(dim=-1).item()
                pred_list.append(pred)
                
                # Get ground truth
                if batch_labels and len(batch_labels) > 0:
                    labels = batch_labels[0]['labels']
                    if task in labels and cell_id - 1 < len(labels[task]):
                        true_list.append(labels[task][cell_id - 1])
                    else:
                        true_list.append(0)
                else:
                    true_list.append(0)
            
            predictions[task] = pred_list
            ground_truth[task] = true_list
    
    return predictions, ground_truth

def calculate_detailed_metrics(all_predictions, all_ground_truth):
    """Calculate detailed metrics including precision, recall, F1"""
    
    metrics = {}
    
    for task in LABEL_MAPS.keys():
        if task not in all_predictions or len(all_predictions[task]) == 0:
            continue
        
        y_true = np.array(all_ground_truth[task])
        y_pred = np.array(all_predictions[task])
        
        # Get unique labels
        unique_labels = sorted(set(y_true) | set(y_pred))
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_labels, average=None, zero_division=0
        )
        
        # Weighted average
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Macro average
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Micro average
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Store metrics
        metrics[task] = {
            'accuracy': accuracy,
            'weighted_avg': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            },
            'macro_avg': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'micro_avg': {
                'precision': precision_micro,
                'recall': recall_micro,
                'f1': f1_micro
            },
            'per_class': {}
        }
        
        # Per-class metrics
        label_names = LABEL_MAPS[task]
        for i, label_id in enumerate(unique_labels):
            if label_id < len(label_names):
                label_name = label_names[label_id]
            else:
                label_name = f"Unknown_{label_id}"
            
            metrics[task]['per_class'][label_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
        
        # Confusion matrix
        metrics[task]['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
    
    return metrics

def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    data_file = sys.argv[2] if len(sys.argv) > 2 else './data/multilabel_taxonomy/test.json'
    
    # Load model
    model, args = load_model()
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    total_tables = len(data)
    data = data[:sample_size]
    print(f"Found {total_tables} tables, evaluating {len(data)} tables")
    
    # Collect all predictions and ground truth
    all_predictions = defaultdict(list)
    all_ground_truth = defaultdict(list)
    
    # Process tables
    print("\nEvaluating tables...")
    start_time = time.time()
    
    for i, table in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)} tables...", end='\r')
        
        predictions, ground_truth = evaluate_table_detailed(model, table, args)
        
        if predictions and ground_truth:
            for task in LABEL_MAPS.keys():
                if task in predictions:
                    all_predictions[task].extend(predictions[task])
                    all_ground_truth[task].extend(ground_truth[task])
    
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(data)} tables in {elapsed:.1f} seconds")
    
    # Calculate metrics
    print("\nCalculating detailed metrics...")
    metrics = calculate_detailed_metrics(all_predictions, all_ground_truth)
    
    # Print results
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)
    
    for task in LABEL_MAPS.keys():
        if task not in metrics:
            continue
        
        print(f"\n{'='*40}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*40}")
        
        task_metrics = metrics[task]
        
        # Overall metrics
        print(f"\n📊 Overall Metrics:")
        print(f"  Accuracy: {task_metrics['accuracy']:.3f}")
        
        print(f"\n📈 Weighted Average (considers class imbalance):")
        print(f"  Precision: {task_metrics['weighted_avg']['precision']:.3f}")
        print(f"  Recall:    {task_metrics['weighted_avg']['recall']:.3f}")
        print(f"  F1-Score:  {task_metrics['weighted_avg']['f1']:.3f}")
        
        print(f"\n📊 Macro Average (unweighted mean):")
        print(f"  Precision: {task_metrics['macro_avg']['precision']:.3f}")
        print(f"  Recall:    {task_metrics['macro_avg']['recall']:.3f}")
        print(f"  F1-Score:  {task_metrics['macro_avg']['f1']:.3f}")
        
        print(f"\n📊 Micro Average (global calculation):")
        print(f"  Precision: {task_metrics['micro_avg']['precision']:.3f}")
        print(f"  Recall:    {task_metrics['micro_avg']['recall']:.3f}")
        print(f"  F1-Score:  {task_metrics['micro_avg']['f1']:.3f}")
        
        # Per-class metrics (top classes)
        print(f"\n📋 Per-Class Metrics:")
        per_class = task_metrics['per_class']
        
        # Sort by support (most common classes first)
        sorted_classes = sorted(per_class.items(), 
                              key=lambda x: x[1]['support'], 
                              reverse=True)
        
        # Show top 5 classes
        for label_name, class_metrics in sorted_classes[:5]:
            p = class_metrics['precision']
            r = class_metrics['recall']
            f1 = class_metrics['f1']
            s = class_metrics['support']
            if s > 0:  # Only show classes with support
                print(f"  {label_name:12} P:{p:.3f} R:{r:.3f} F1:{f1:.3f} Support:{s}")
        
        if len(sorted_classes) > 5:
            print(f"  ... and {len(sorted_classes) - 5} more classes")
        
        # Sample counts
        print(f"\n📊 Total samples evaluated: {len(all_predictions[task])}")
    
    # Save results
    output_file = f"./models/detailed_metrics_{sample_size}.json"
    with open(output_file, 'w') as f:
        # Convert numpy values to native Python types
        json_metrics = {}
        for task, task_metrics in metrics.items():
            json_metrics[task] = {
                'accuracy': float(task_metrics['accuracy']),
                'weighted_avg': {
                    'precision': float(task_metrics['weighted_avg']['precision']),
                    'recall': float(task_metrics['weighted_avg']['recall']),
                    'f1': float(task_metrics['weighted_avg']['f1'])
                },
                'macro_avg': {
                    'precision': float(task_metrics['macro_avg']['precision']),
                    'recall': float(task_metrics['macro_avg']['recall']),
                    'f1': float(task_metrics['macro_avg']['f1'])
                },
                'micro_avg': {
                    'precision': float(task_metrics['micro_avg']['precision']),
                    'recall': float(task_metrics['micro_avg']['recall']),
                    'f1': float(task_metrics['micro_avg']['f1'])
                },
                'per_class': {}
            }
            
            # Add per-class metrics
            if 'per_class' in task_metrics:
                for label_name, class_metrics in task_metrics['per_class'].items():
                    json_metrics[task]['per_class'][label_name] = {
                        'precision': float(class_metrics['precision']),
                        'recall': float(class_metrics['recall']),
                        'f1': float(class_metrics['f1']),
                        'support': int(class_metrics['support'])
                    }
        json.dump(json_metrics, f, indent=2)
    
    print(f"\n✓ Detailed metrics saved to: {output_file}")

if __name__ == '__main__':
    main()