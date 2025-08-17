#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare Korean financial audit data from taxonomy directory for multi-label training.
Processes JSON files containing Korean financial tables with multi-label annotations.
"""

import json
import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import logging

from dataset_types import LABEL_MAPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_json_line(line: str) -> Dict:
    """Parse a single JSON line from taxonomy files."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON line: {e}")
        return None


def process_taxonomy_file(filepath: Path) -> List[Dict]:
    """Process a single taxonomy JSON file and extract tables with labels."""
    tables = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Try to parse as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        # Each item might be a JSON string itself
                        table_data = json.loads(item)
                        if table_data and 'cells' in table_data:
                            tables.append(table_data)
                    elif isinstance(item, dict) and 'cells' in item:
                        tables.append(item)
            elif isinstance(data, dict) and 'cells' in data:
                tables.append(data)
        except json.JSONDecodeError:
            # Try line-by-line parsing
            lines = content.strip().split('\n')
            for line in lines:
                if line.strip():
                    if line.startswith('[') and line.endswith(']'):
                        # It's a JSON array on a single line
                        try:
                            items = json.loads(line)
                            for item in items:
                                if isinstance(item, str):
                                    table_data = json.loads(item)
                                    if table_data and 'cells' in table_data:
                                        tables.append(table_data)
                                elif isinstance(item, dict) and 'cells' in item:
                                    tables.append(item)
                        except:
                            pass
                    else:
                        # Try parsing as single JSON object
                        table_data = parse_json_line(line)
                        if table_data and 'cells' in table_data:
                            tables.append(table_data)
                            
        logger.info(f"Processed {filepath.name}: found {len(tables)} tables")
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        
    return tables


def extract_company_info(filename: str) -> Dict[str, str]:
    """Extract company name, period, and type from filename."""
    # Format: 회사명_audit_YYYY_MM_[con|sep]_body.json
    parts = filename.replace('.json', '').split('_')
    
    info = {
        'company': parts[0] if len(parts) > 0 else 'unknown',
        'year': parts[2] if len(parts) > 2 else 'unknown',
        'month': parts[3] if len(parts) > 3 else 'unknown',
        'type': parts[4] if len(parts) > 4 else 'unknown'  # con or sep
    }
    
    return info


def convert_to_training_format(table_data: Dict, metadata: Dict = None) -> Dict:
    """Convert taxonomy table format to training format."""
    cells = table_data.get('cells', {})
    
    # Extract table structure
    rows = set()
    cols = set()
    cell_list = []
    
    for cell_ref, cell_data in cells.items():
        # Handle merged cells (e.g., "A1:B2")
        if ':' in cell_ref:
            start, end = cell_ref.split(':')
            # Extract start row/col
            col_start = ''.join(c for c in start if c.isalpha())
            row_start = int(''.join(c for c in start if c.isdigit()))
            # Extract end row/col
            col_end = ''.join(c for c in end if c.isalpha())
            row_end = int(''.join(c for c in end if c.isdigit()))
            
            rows.add(row_start)
            rows.add(row_end)
            cols.add(col_start)
            cols.add(col_end)
            
            # Store merged cell info
            cell_info = {
                'position': cell_ref,
                'row_start': row_start,
                'row_end': row_end,
                'col_start': col_start,
                'col_end': col_end,
                'is_merged': True,
                'value': cell_data.get('value', ''),
                'labels': {
                    'fiscal_year': cell_data.get('fiscal_year', 'UNK'),
                    'period': cell_data.get('period', 'UNK'),
                    'qa': cell_data.get('qa', 'UNK'),
                    'decimal': cell_data.get('decimal', 'UNK'),
                    'unit': cell_data.get('unit', 'UNK'),
                    'cell_type': cell_data.get('cell_type', 'other')
                },
                'style': cell_data.get('style', {})
            }
        else:
            # Single cell
            col = ''.join(c for c in cell_ref if c.isalpha())
            row = int(''.join(c for c in cell_ref if c.isdigit()))
            
            rows.add(row)
            cols.add(col)
            
            cell_info = {
                'position': cell_ref,
                'row': row,
                'col': col,
                'is_merged': False,
                'value': cell_data.get('value', ''),
                'labels': {
                    'fiscal_year': cell_data.get('fiscal_year', 'UNK'),
                    'period': cell_data.get('period', 'UNK'),
                    'qa': cell_data.get('qa', 'UNK'),
                    'decimal': cell_data.get('decimal', 'UNK'),
                    'unit': cell_data.get('unit', 'UNK'),
                    'cell_type': cell_data.get('cell_type', 'other')
                },
                'style': cell_data.get('style', {})
            }
        
        cell_list.append(cell_info)
    
    # Create training instance
    training_instance = {
        'table_id': f"{metadata.get('company', 'unknown')}_{metadata.get('year', '')}_{metadata.get('month', '')}_{metadata.get('type', '')}",
        'cells': cell_list,
        'metadata': metadata
    }
    
    return training_instance


def validate_labels(instance: Dict) -> bool:
    """Validate that all labels are valid according to LABEL_MAPS."""
    for cell in instance.get('cells', []):
        labels = cell.get('labels', {})
        for task, label in labels.items():
            if task in LABEL_MAPS:
                if label not in LABEL_MAPS[task]:
                    logger.warning(f"Invalid label {label} for task {task}")
                    # Try to fix common issues
                    if label in ['', None, 'null']:
                        labels[task] = 'UNK'
                    else:
                        # Default to UNK for invalid labels
                        labels[task] = 'UNK'
    return True


def analyze_dataset(instances: List[Dict]):
    """Analyze the dataset and print statistics."""
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    
    # Count instances and cells
    total_cells = sum(len(inst['cells']) for inst in instances)
    print(f"\nTotal tables: {len(instances)}")
    print(f"Total cells: {total_cells}")
    print(f"Average cells per table: {total_cells/len(instances):.1f}")
    
    # Count companies
    companies = Counter(inst['metadata']['company'] for inst in instances)
    print(f"\nUnique companies: {len(companies)}")
    print("Top 10 companies:")
    for company, count in companies.most_common(10):
        print(f"  {company}: {count} tables")
    
    # Count periods
    periods = Counter(f"{inst['metadata']['year']}_{inst['metadata']['month']}" 
                     for inst in instances)
    print(f"\nPeriods covered:")
    for period, count in sorted(periods.items()):
        print(f"  {period}: {count} tables")
    
    # Count types
    types = Counter(inst['metadata']['type'] for inst in instances)
    print(f"\nTable types:")
    for type_name, count in types.items():
        print(f"  {type_name}: {count} tables")
    
    # Analyze label distributions
    print("\n" + "-"*50)
    print("Label Distributions")
    print("-"*50)
    
    for task in LABEL_MAPS.keys():
        label_counts = Counter()
        for inst in instances:
            for cell in inst['cells']:
                label = cell['labels'].get(task, 'UNK')
                label_counts[label] += 1
        
        print(f"\n{task}:")
        total = sum(label_counts.values())
        for label in LABEL_MAPS[task]:
            count = label_counts.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {label:15s}: {count:8d} ({percentage:5.1f}%)")


def split_dataset(instances: List[Dict], 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  stratify_by: str = 'company') -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test sets with optional stratification."""
    
    if stratify_by == 'company':
        # Group by company to ensure each company appears in all splits
        company_groups = defaultdict(list)
        for inst in instances:
            company = inst['metadata']['company']
            company_groups[company].append(inst)
        
        train, val, test = [], [], []
        
        for company, company_instances in company_groups.items():
            # Shuffle instances for this company
            random.shuffle(company_instances)
            
            n = len(company_instances)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train.extend(company_instances[:n_train])
            val.extend(company_instances[n_train:n_train+n_val])
            test.extend(company_instances[n_train+n_val:])
            
    else:
        # Simple random split
        random.shuffle(instances)
        
        n = len(instances)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train = instances[:n_train]
        val = instances[n_train:n_train+n_val]
        test = instances[n_train+n_val:]
    
    return train, val, test


def save_splits(train: List[Dict], val: List[Dict], test: List[Dict], output_dir: Path):
    """Save dataset splits to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        output_file = output_dir / f'{split_name}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(split_data)} instances to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare taxonomy data for multi-label training')
    parser.add_argument('--input_dir', type=str, default='./data/taxonomy',
                       help='Directory containing taxonomy JSON files')
    parser.add_argument('--output_dir', type=str, default='./data/multilabel_taxonomy',
                       help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--stratify_by', type=str, default='company',
                       choices=['company', 'none'],
                       help='Stratification strategy for splitting')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze and print dataset statistics')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Process taxonomy files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    json_files = sorted(input_dir.glob('*.json'))
    
    if args.max_files:
        json_files = json_files[:args.max_files]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process all files and collect instances
    all_instances = []
    
    for json_file in json_files:
        # Extract metadata from filename
        metadata = extract_company_info(json_file.name)
        
        # Process file
        tables = process_taxonomy_file(json_file)
        
        # Convert each table to training format
        for table in tables:
            instance = convert_to_training_format(table, metadata)
            if validate_labels(instance):
                all_instances.append(instance)
    
    print(f"\nTotal training instances created: {len(all_instances)}")
    
    if len(all_instances) == 0:
        logger.error("No valid training instances found")
        return
    
    # Analyze dataset if requested
    if args.analyze:
        analyze_dataset(all_instances)
    
    # Split dataset
    train, val, test = split_dataset(
        all_instances,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_by=args.stratify_by
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train)} instances")
    print(f"  Val: {len(val)} instances")
    print(f"  Test: {len(test)} instances")
    
    # Save splits
    output_dir = Path(args.output_dir)
    save_splits(train, val, test, output_dir)
    
    print(f"\nData preparation complete! Files saved to {output_dir}")


if __name__ == '__main__':
    main()