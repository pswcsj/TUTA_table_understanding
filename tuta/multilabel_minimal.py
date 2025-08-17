#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Multi-label TUTA Extension
Only the essential components - no redundant main() or argument parsing
Use with: python train.py --target multilabel --dataset_paths ...
"""

import json
import random
import torch
import torch.nn as nn
import argparse

# Import original TUTA components
import model.heads as hds
import model.pretrains as ptm
import trainers as tnr
import dynamic_data as dymdata
from ctc_finetune import TctcReader, TctcTok
from dataset_types import LABEL_MAPS
import utils as ut
import time


# ============ 1. MODEL COMPONENTS ============


class MultiLabelCtcHead(hds.CtcHead):
    """Multi-label classification head extending original CtcHead"""

    def __init__(self, config):
        config.num_ctc_type = 2  # Dummy value for parent init
        super().__init__(config)

        # Remove original predict_linear
        del self.predict_linear

        # Add multi-label classifiers
        self.classifiers = nn.ModuleDict(
            {
                task: nn.Linear(config.hidden_size, len(labels))
                for task, labels in LABEL_MAPS.items()
            }
        )

        # Loss functions for each task
        self.loss_fns = {
            task: nn.CrossEntropyLoss(ignore_index=-1) for task in LABEL_MAPS.keys()
        }

    def forward(self, encoded_states, indicator, batch_labels):
        # Get cell encodings using parent's aggregation
        # This creates one state per unique indicator value
        cell_states = self.aggr_funcs[self.aggregator](encoded_states, indicator)

        # Collect all valid cells and labels across the batch
        all_cell_features = []
        all_labels = {task: [] for task in LABEL_MAPS.keys()}

        for batch_item in batch_labels:
            batch_idx = batch_item["batch_idx"]
            item_labels = batch_item["labels"]

            batch_indicator = indicator[batch_idx]
            batch_cell_states = cell_states[batch_idx]

            # Get unique cell IDs for this batch item (excluding -1 for CLS/SEP and 0 for PAD)
            unique_indicators, inverse_indices = torch.unique(
                batch_indicator, sorted=True, return_inverse=True
            )
            valid_mask = unique_indicators > 0
            valid_indicators = unique_indicators[valid_mask]

            if len(valid_indicators) == 0:
                continue

            # Get the cell states for valid indicators
            valid_cell_states = []
            valid_cell_labels = {task: [] for task in LABEL_MAPS.keys()}

            for cell_id in valid_indicators:
                # Find first occurrence of this cell_id
                cell_idx = (batch_indicator == cell_id).nonzero(as_tuple=True)[0][0]
                valid_cell_states.append(batch_cell_states[cell_idx])

                # Get label for this cell (cell_id starts from 1, so use cell_id - 1 as index)
                label_idx = cell_id.item() - 1
                for task in LABEL_MAPS.keys():
                    if task in item_labels and label_idx < len(item_labels[task]):
                        valid_cell_labels[task].append(item_labels[task][label_idx])

            if len(valid_cell_states) > 0:
                valid_cell_states = torch.stack(valid_cell_states)
                all_cell_features.append(valid_cell_states)

                # Add valid labels
                for task in LABEL_MAPS.keys():
                    all_labels[task].extend(valid_cell_labels[task])

        if len(all_cell_features) == 0:
            zero_loss = torch.tensor(
                0.0, requires_grad=True, device=encoded_states.device
            )
            zero_tensor = torch.tensor(0.0, device=encoded_states.device)
            return {
                task: (zero_loss, zero_tensor, zero_tensor)
                for task in LABEL_MAPS.keys()
            }

        # Concatenate all cell features
        all_cell_features = torch.cat(all_cell_features, dim=0)

        # Apply transformation
        cell_features = self.uniform_linear(all_cell_features)
        cell_features = self.tanh(cell_features)

        # Compute predictions for each task
        results = {}
        for task, classifier in self.classifiers.items():
            logits = classifier(cell_features)

            if task in all_labels and len(all_labels[task]) > 0:
                labels = torch.tensor(all_labels[task], device=logits.device)

                if len(labels) == cell_features.size(0):
                    loss = self.loss_fns[task](logits, labels)
                    predictions = logits.argmax(dim=-1)
                    correct = torch.sum(predictions.eq(labels).float())
                    count = torch.tensor(float(len(labels)), device=logits.device)
                else:
                    # Size mismatch - this shouldn't happen with proper mapping
                    print(
                        f"ERROR {task}: Size mismatch! cells={cell_features.size(0)}, labels={len(labels)}"
                    )
                    loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
                    correct = torch.tensor(0.0, device=logits.device)
                    count = torch.tensor(1.0, device=logits.device)
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
                correct = torch.tensor(0.0, device=logits.device)
                count = torch.tensor(1.0, device=logits.device)

            results[task] = (loss, correct, count)

        return results


class TUTAMultiLabelNative(ptm.TUTAforCTC):
    """Native TUTA model for multi-label classification"""

    def __init__(self, config):
        # Store the original target and use 'tuta' for backbone
        original_target = config.target
        config.target = "tuta"  # Use tuta backbone
        config.num_ctc_type = 2  # Dummy value
        super().__init__(config)
        config.target = original_target  # Restore original target

        # Replace CTC head with multi-label head
        del self.ctc_head
        self.multi_label_head = MultiLabelCtcHead(config)

    def forward(
        self,
        token_id,
        num_mag,
        num_pre,
        num_top,
        num_low,
        token_order,
        pos_row,
        pos_col,
        pos_top,
        pos_left,
        format_vec,
        indicator,
        batch_labels,
    ):
        # Get encoded states from backbone
        encoded_states = self.backbone(
            token_id,
            num_mag,
            num_pre,
            num_top,
            num_low,
            token_order,
            pos_row,
            pos_col,
            pos_top,
            pos_left,
            format_vec,
            indicator,
        )

        # Apply multi-label head
        return self.multi_label_head(encoded_states, indicator, batch_labels)


# ============ 2. DATA COMPONENTS ============


class MultiLabelTableReader(TctcReader):
    """Minimal extension of TctcReader for multi-label"""

    def get_label_matrix(self, cell_matrix, task):
        """Extract label matrix for a specific task"""
        label_matrix = []
        for row in cell_matrix:
            label_row = []
            for cell in row:
                if isinstance(cell, dict):
                    if "Labels" in cell:
                        label = cell["Labels"].get(task, "UNK")
                    else:
                        label = cell.get(task, "UNK")

                    # Convert to index
                    if label in LABEL_MAPS[task]:
                        label_idx = LABEL_MAPS[task].index(label)
                    else:
                        label_idx = 0  # UNK
                else:
                    label_idx = 0

                label_row.append(label_idx)
            label_matrix.append(label_row)

        return label_matrix


class MultiLabelTableTokenizer(TctcTok):
    """Minimal extension of TctcTok for multi-label"""

    def __init__(self, args):
        super().__init__(args)
        self.target = args.target

    def tokenize_multilabel(
        self,
        string_matrix,
        top_position_list,
        left_position_list,
        format_matrix,
        label_matrices,
        header_info,
        root_context="",
    ):
        # Use parent's tokenization with no sampling
        sampling_mask = self.no_sampling(string_matrix)

        # Initialize sequence
        token_list, num_list, pos_list, format_list, ind_list, _, cell_num, seq_len = (
            self.init_table_seq(root_context)
        )

        # Collect labels for cells that are actually tokenized
        cell_labels = {task: [] for task in LABEL_MAPS.keys()}

        # Process cells (simplified version)
        header_rows, header_columns = header_info
        icell = 0

        for irow, string_row in enumerate(string_matrix):
            for icol, cell_string in enumerate(string_row):
                if sampling_mask[irow][icol] == 0:
                    continue

                if not cell_string or cell_string.strip() == "":
                    continue

                # Tokenize cell
                cell_tokens, cell_numbers = self.tokenize_text(
                    cell_string, add_separate=True
                )

                if len(cell_tokens) == 0:
                    continue

                token_list.append(cell_tokens)
                num_list.append(cell_numbers)

                # Handle positions - always use TUTA format (4 values)
                if (
                    irow < header_rows
                    and icol < header_columns
                    and icell < len(top_position_list)
                ):
                    pos_list.append(
                        (
                            irow,
                            icol,
                            top_position_list[icell],
                            left_position_list[icell],
                        )
                    )
                    icell += 1
                else:
                    # Default positions for cells outside header
                    # Use 4 elements to match tree_depth=4
                    pos_list.append(
                        (irow, icol, [0, 0, 0, 0], [0, 0, 0, 0])
                    )  # Use 0 instead of -1

                format_list.append(format_matrix[irow][icol])
                ind_list.append([cell_num] * len(cell_tokens))

                # Collect label for this cell
                for task in LABEL_MAPS.keys():
                    label = label_matrices[task][irow][icol]
                    cell_labels[task].append(label)

                cell_num += 1
                seq_len += len(cell_tokens)

                if seq_len >= self.max_seq_len - 1:
                    break

            if seq_len >= self.max_seq_len - 1:
                break

        # Add SEP token
        token_list.append([1])  # SEP_ID
        num_list.append([self.wordpiece_tokenizer.default_num])
        pos_list.append(pos_list[0] if pos_list else (0, 0, [0, 0, 0, 0], [0, 0, 0, 0]))
        format_list.append(self.default_format)
        ind_list.append([-1])

        # Flatten everything
        return self._flatten_sequences(
            token_list, num_list, pos_list, format_list, ind_list, cell_labels
        )

    def _flatten_sequences(
        self, token_list, num_list, pos_list, format_list, ind_list, cell_labels
    ):
        """Helper to flatten sequences"""
        flat_tokens = []
        flat_nums = []
        flat_indicators = []
        flat_formats = []
        flat_pos_row = []
        flat_pos_col = []
        flat_pos_top = []
        flat_pos_left = []

        for tokens, nums, pos, fmt, inds in zip(
            token_list, num_list, pos_list, format_list, ind_list
        ):
            cell_len = len(tokens)
            flat_tokens.extend(tokens)
            flat_nums.extend(nums)
            flat_indicators.extend(inds)

            # Always use TUTA format (4 values)
            row, col, top, left = pos
            flat_pos_row.extend([row] * cell_len)
            flat_pos_col.extend([col] * cell_len)
            flat_pos_top.extend([top] * cell_len)
            flat_pos_left.extend([left] * cell_len)

            flat_formats.extend([fmt] * cell_len)

        # Labels are already collected per cell in cell_labels

        # Always return TUTA format (9 values)
        return (
            flat_tokens,
            flat_nums,
            flat_pos_row,
            flat_pos_col,
            flat_pos_top,
            flat_pos_left,
            flat_formats,
            flat_indicators,
            cell_labels,
        )


# ============ 3. TRAINER ============


def validate_multilabel(args, gpu_id, rank, val_loader, model):
    """Validation function for multi-label tasks"""
    model.eval()

    total_loss = 0.0
    task_losses = {task: 0.0 for task in LABEL_MAPS.keys()}
    task_correct = {task: 0.0 for task in LABEL_MAPS.keys()}
    task_total = {task: 0.0 for task in LABEL_MAPS.keys()}

    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch (handle both TUTA and base formats)
            if len(batch) == 9:  # TUTA format
                (
                    token_id,
                    num_matrix,
                    pos_row,
                    pos_col,
                    pos_top,
                    pos_left,
                    format_vec,
                    indicator,
                    batch_labels,
                ) = batch
            else:  # Base format
                (
                    token_id,
                    num_matrix,
                    pos_top,
                    pos_left,
                    format_vec,
                    indicator,
                    batch_labels,
                ) = batch
                pos_row = pos_col = None

            # Extract numerical features
            num_mag = num_matrix[:, :, 0]
            num_pre = num_matrix[:, :, 1]
            num_top_val = num_matrix[:, :, 2]
            num_low = num_matrix[:, :, 3]
            token_order = torch.zeros_like(token_id)

            # Move to device
            if gpu_id is not None:
                token_id = token_id.cuda(gpu_id)
                num_mag = num_mag.cuda(gpu_id)
                num_pre = num_pre.cuda(gpu_id)
                num_top_val = num_top_val.cuda(gpu_id)
                num_low = num_low.cuda(gpu_id)
                token_order = token_order.cuda(gpu_id)
                if pos_row is not None:
                    pos_row = pos_row.cuda(gpu_id)
                    pos_col = pos_col.cuda(gpu_id)
                pos_top = pos_top.cuda(gpu_id)
                pos_left = pos_left.cuda(gpu_id)
                format_vec = format_vec.cuda(gpu_id)
                indicator = indicator.cuda(gpu_id)

            # Forward pass
            if pos_row is not None:
                results = model(
                    token_id,
                    num_mag,
                    num_pre,
                    num_top_val,
                    num_low,
                    token_order,
                    pos_row,
                    pos_col,
                    pos_top,
                    pos_left,
                    format_vec,
                    indicator,
                    batch_labels,
                )
            else:
                results = model(
                    token_id,
                    num_mag,
                    num_pre,
                    num_top_val,
                    num_low,
                    token_order,
                    pos_top,
                    pos_left,
                    format_vec,
                    indicator,
                    batch_labels,
                )

            # Accumulate metrics
            batch_loss = 0.0
            for task, (task_loss, correct, count) in results.items():
                batch_loss += task_loss.item()
                task_losses[task] += task_loss.item()
                task_correct[task] += correct.item()
                task_total[task] += count.item()

            total_loss += batch_loss
            num_batches += 1

    # Calculate averages
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        task_avg_losses = {
            task: loss / num_batches for task, loss in task_losses.items()
        }
        task_accuracies = {
            task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
            for task in LABEL_MAPS.keys()
        }
    else:
        avg_loss = 0.0
        task_avg_losses = {task: 0.0 for task in LABEL_MAPS.keys()}
        task_accuracies = {task: 0.0 for task in LABEL_MAPS.keys()}

    model.train()
    return avg_loss, task_avg_losses, task_accuracies


def train_multilabel(args, gpu_id, rank, loader, model, optimizer, scheduler):
    """Training function for multi-label tasks"""
    model.train()
    start_time = time.time()

    total_loss = 0.0
    task_losses = {task: 0.0 for task in LABEL_MAPS.keys()}
    task_correct = {task: 0.0 for task in LABEL_MAPS.keys()}
    task_total = {task: 0.0 for task in LABEL_MAPS.keys()}

    # Load validation data if provided
    val_loader = None
    if hasattr(args, "val_dataset_paths") and args.val_dataset_paths:
        val_args = argparse.Namespace(**vars(args))
        val_args.dataset_paths = args.val_dataset_paths
        if args.dist_train:
            val_loader = MultiLabelDataLoader(
                val_args, rank, args.world_size, do_shuffle=False
            )
        else:
            val_loader = MultiLabelDataLoader(val_args, 0, 1, do_shuffle=False)
        print(f"Loaded validation dataset from: {args.val_dataset_paths}")

    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break

        try:
            batch = next(loader_iter)
        except StopIteration:
            # Dataset exhausted, create new iterator to continue training
            loader_iter = iter(loader)
            batch = next(loader_iter)

        # Unpack batch (handle both TUTA and base formats)
        if len(batch) == 9:  # TUTA format
            (
                token_id,
                num_matrix,
                pos_row,
                pos_col,
                pos_top,
                pos_left,
                format_vec,
                indicator,
                batch_labels,
            ) = batch
        else:  # Base format
            (
                token_id,
                num_matrix,
                pos_top,
                pos_left,
                format_vec,
                indicator,
                batch_labels,
            ) = batch
            pos_row = pos_col = None

        # Extract numerical features
        num_mag = num_matrix[:, :, 0]
        num_pre = num_matrix[:, :, 1]
        num_top = num_matrix[:, :, 2]
        num_low = num_matrix[:, :, 3]
        # token_order represents position within cell, not within sequence
        # For simplicity, use zeros as we're not using cell-level ordering
        token_order = torch.zeros_like(token_id)

        # Move to device
        model.zero_grad()
        if gpu_id is not None:
            token_id = token_id.cuda(gpu_id)
            num_mag = num_mag.cuda(gpu_id)
            num_pre = num_pre.cuda(gpu_id)
            num_top = num_top.cuda(gpu_id)
            num_low = num_low.cuda(gpu_id)
            token_order = token_order.cuda(gpu_id)
            if pos_row is not None:
                pos_row = pos_row.cuda(gpu_id)
                pos_col = pos_col.cuda(gpu_id)
            pos_top = pos_top.cuda(gpu_id)
            pos_left = pos_left.cuda(gpu_id)
            format_vec = format_vec.cuda(gpu_id)
            indicator = indicator.cuda(gpu_id)

        # Forward pass
        if pos_row is not None:
            results = model(
                token_id,
                num_mag,
                num_pre,
                num_top,
                num_low,
                token_order,
                pos_row,
                pos_col,
                pos_top,
                pos_left,
                format_vec,
                indicator,
                batch_labels,
            )
        else:
            # For base model without pos_row/pos_col
            results = model(
                token_id,
                num_mag,
                num_pre,
                num_top,
                num_low,
                token_order,
                pos_top,
                pos_left,
                format_vec,
                indicator,
                batch_labels,
            )

        # Compute loss
        loss = torch.tensor(0.0, device=token_id.device)
        for task, (task_loss, correct, count) in results.items():
            loss = loss + task_loss
            task_losses[task] += task_loss.item()
            task_correct[task] += correct.item()
            task_total[task] += count.item()

        # Backward
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Report progress (only from rank 0 to avoid duplicate output)
        if steps % args.report_steps == 0:
            # Only print from rank 0 (master process)
            is_dist = hasattr(args, "dist_train") and args.dist_train
            if not is_dist or (is_dist and rank == 0):
                elapsed = time.time() - start_time
                done_tokens = args.batch_size * token_id.size(1) * args.report_steps

                # In distributed training, multiply by world_size for actual throughput
                if is_dist and hasattr(args, "world_size"):
                    done_tokens *= args.world_size

                print(
                    f"| {steps:8d}/{total_steps:8d} steps | {done_tokens/elapsed:8.2f} tokens/s | loss {total_loss/args.report_steps:7.2f}"
                )

                for task in LABEL_MAPS.keys():
                    if task_total[task] > 0:
                        print(
                            f"  {task}: loss={task_losses[task]/args.report_steps:.4f}, acc={task_correct[task]/task_total[task]:.3f}"
                        )

                # Run validation if available and at validation interval
                validation_interval = getattr(args, "validation_steps", 1000)
                if val_loader and steps % validation_interval == 0:
                    print("\nRunning validation...")
                    val_loss, val_task_losses, val_task_accs = validate_multilabel(
                        args, gpu_id, rank, val_loader, model
                    )
                    print(f"Validation loss: {val_loss:.4f}")
                    for task in LABEL_MAPS.keys():
                        print(
                            f"  Val {task}: loss={val_task_losses[task]:.4f}, acc={val_task_accs[task]:.3f}"
                        )
                    print()

            # Reset counters (all processes)
            total_loss = 0.0
            task_losses = {task: 0.0 for task in LABEL_MAPS.keys()}
            task_correct = {task: 0.0 for task in LABEL_MAPS.keys()}
            task_total = {task: 0.0 for task in LABEL_MAPS.keys()}
            start_time = time.time()

        # Save checkpoint (only from rank 0 to avoid conflicts)
        if steps % args.save_checkpoint_steps == 0:
            is_dist = hasattr(args, "dist_train") and args.dist_train
            if not is_dist or (is_dist and rank == 0):
                ut.save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1

    # Save final model after training completes (only from rank 0)
    is_dist = hasattr(args, "dist_train") and args.dist_train
    if not is_dist or (is_dist and rank == 0):
        final_path = args.output_model_path + "-final"
        ut.save_model(model, final_path)
        print(f"\nFinal model saved to: {final_path}")


# ============ 4. DATA LOADER ============


def convert_to_hierarchical(data):
    """Convert flat JSON format to hierarchical format expected by TUTA"""
    cells = data.get("cells", [])
    max_row = 0
    max_col = 0
    merged_regions = []
    cell_data = {}

    def col_letter_to_num(col_str):
        num = 0
        for c in col_str:
            num = num * 26 + (ord(c.upper()) - ord("A") + 1)
        return num - 1

    for cell_info in cells:
        if cell_info.get("is_merged", False):
            row_start = cell_info.get("row_start", 0)
            row_end = cell_info.get("row_end", row_start)
            col_start = col_letter_to_num(cell_info.get("col_start", "A"))
            col_end = col_letter_to_num(
                cell_info.get("col_end", cell_info.get("col_start", "A"))
            )

            max_row = max(max_row, row_end)
            max_col = max(max_col, col_end + 1)
            merged_regions.append(
                {
                    "FirstRow": row_start - 1,
                    "FirstColumn": col_start,
                    "LastRow": row_end - 1,
                    "LastColumn": col_end,
                }
            )
            cell_data[(row_start - 1, col_start)] = cell_info
        else:
            row = cell_info.get("row", 1) - 1
            col = col_letter_to_num(cell_info.get("col", "A"))
            max_row = max(max_row, row + 1)
            max_col = max(max_col, col + 1)
            cell_data[(row, col)] = cell_info

    # Initialize cell matrix
    cell_matrix = []
    for r in range(max_row):
        row = []
        for c in range(max_col):
            if (r, c) in cell_data:
                cell_info = cell_data[(r, c)]
                labels = cell_info.get("labels", {})
                style = cell_info.get("style", {})
                row.append(
                    {
                        "V": str(cell_info.get("value", "")),
                        "Labels": {
                            task: labels.get(task, "UNK") for task in LABEL_MAPS.keys()
                        },
                        "TB": 1 if style.get("border", False) else 0,
                        "BB": 1 if style.get("border", False) else 0,
                        "LB": 1 if style.get("border", False) else 0,
                        "RB": 1 if style.get("border", False) else 0,
                        "DT": 0,
                        "HF": 0,
                        "FB": 1 if style.get("bold", False) else 0,
                        "BC": "#ffffff",
                        "FC": "#000000",
                    }
                )
            else:
                row.append(
                    {
                        "V": "",
                        "Labels": {task: "UNK" for task in LABEL_MAPS.keys()},
                        "TB": 0,
                        "BB": 0,
                        "LB": 0,
                        "RB": 0,
                        "DT": 0,
                        "HF": 0,
                        "FB": 0,
                        "BC": "#ffffff",
                        "FC": "#000000",
                    }
                )
        cell_matrix.append(row)

    return {
        "Cells": cell_matrix,
        "MergedRegions": merged_regions,
        "TopTreeRoot": None,
        "LeftTreeRoot": None,
        "TopHeaderRowsNumber": 1,
        "LeftHeaderColumnsNumber": 0,
        "TableId": data.get("table_id", "unknown"),
    }


def collate_multilabel(batch):
    """Collate function for multi-label batches"""
    if len(batch[0]) == 9:  # TUTA format
        (
            tokens,
            nums,
            pos_rows,
            pos_cols,
            pos_tops,
            pos_lefts,
            formats,
            indicators,
            labels,
        ) = zip(*batch)

        # Find max length
        max_len = max(len(t) for t in tokens)

        # Pad everything
        padded_tokens = []
        padded_nums = []
        padded_pos_rows = []
        padded_pos_cols = []
        padded_pos_tops = []
        padded_pos_lefts = []
        padded_formats = []
        padded_indicators = []

        for i in range(len(batch)):
            pad_len = max_len - len(tokens[i])
            padded_tokens.append(tokens[i] + [0] * pad_len)
            padded_nums.append(nums[i] + [[0, 0, 0, 0]] * pad_len)
            padded_pos_rows.append(pos_rows[i] + [0] * pad_len)
            padded_pos_cols.append(pos_cols[i] + [0] * pad_len)
            # Determine position length from first sample
            pos_len = len(pos_tops[i][0]) if len(pos_tops[i]) > 0 else 4
            padded_pos_tops.append(pos_tops[i] + [[0] * pos_len] * pad_len)
            padded_pos_lefts.append(pos_lefts[i] + [[0] * pos_len] * pad_len)
            padded_formats.append(formats[i] + [[0] * 11] * pad_len)
            padded_indicators.append(indicators[i] + [-1] * pad_len)

        # Keep labels separated by batch item with batch indices
        batch_labels = []
        for i, label_dict in enumerate(labels):
            batch_labels.append({"batch_idx": i, "labels": label_dict})

        return (
            torch.tensor(padded_tokens),
            torch.tensor(padded_nums),
            torch.tensor(padded_pos_rows),
            torch.tensor(padded_pos_cols),
            torch.tensor(padded_pos_tops),
            torch.tensor(padded_pos_lefts),
            torch.tensor(padded_formats, dtype=torch.float),
            torch.tensor(padded_indicators),
            batch_labels,
        )
    else:  # Base format
        tokens, nums, pos_tops, pos_lefts, formats, indicators, labels = zip(*batch)
        max_len = max(len(t) for t in tokens)

        padded_tokens = []
        padded_nums = []
        padded_pos_tops = []
        padded_pos_lefts = []
        padded_formats = []
        padded_indicators = []

        for i in range(len(batch)):
            pad_len = max_len - len(tokens[i])
            padded_tokens.append(tokens[i] + [0] * pad_len)
            padded_nums.append(nums[i] + [[0, 0, 0, 0]] * pad_len)
            # Determine position length from first sample
            pos_len = len(pos_tops[i][0]) if len(pos_tops[i]) > 0 else 4
            padded_pos_tops.append(pos_tops[i] + [[0] * pos_len] * pad_len)
            padded_pos_lefts.append(pos_lefts[i] + [[0] * pos_len] * pad_len)
            padded_formats.append(formats[i] + [[0] * 11] * pad_len)
            padded_indicators.append(indicators[i] + [-1] * pad_len)

        # Keep labels separated by batch item with batch indices
        batch_labels = []
        for i, label_dict in enumerate(labels):
            batch_labels.append({"batch_idx": i, "labels": label_dict})

        return (
            torch.tensor(padded_tokens),
            torch.tensor(padded_nums),
            torch.tensor(padded_pos_tops),
            torch.tensor(padded_pos_lefts),
            torch.tensor(padded_formats, dtype=torch.float),
            torch.tensor(padded_indicators),
            batch_labels,
        )


class MultiLabelDataLoader:
    """DataLoader for multi-label classification"""

    def __init__(self, args, proc_id=0, proc_num=1, do_shuffle=True):
        self.args = args
        self.batch_size = args.batch_size
        self.reader = MultiLabelTableReader(args)
        self.tokenizer = MultiLabelTableTokenizer(args)

        # Load data from dataset_paths (compatible with train.py)
        self.samples = []
        if hasattr(args, "dataset_paths"):
            data_paths = (
                args.dataset_paths
                if isinstance(args.dataset_paths, list)
                else [args.dataset_paths]
            )
        elif hasattr(args, "train_path"):
            data_paths = [args.train_path]
        else:
            raise ValueError("No data path specified")

        for data_path in data_paths:
            with open(data_path, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]

                for table_data in data:
                    if "cells" in table_data:
                        hier_table = convert_to_hierarchical(table_data)
                    else:
                        hier_table = table_data

                    result = self.reader.get_inputs(hier_table)
                    if result is not None:
                        string_matrix, position_lists, header_info, format_matrix = (
                            result
                        )
                        top_position_list, left_position_list = position_lists

                        label_matrices = {}
                        for task in LABEL_MAPS.keys():
                            label_matrices[task] = self.reader.get_label_matrix(
                                hier_table["Cells"], task
                            )

                        tokenized = self.tokenizer.tokenize_multilabel(
                            string_matrix,
                            top_position_list,
                            left_position_list,
                            format_matrix,
                            label_matrices,
                            header_info,
                        )

                        if tokenized is not None:
                            self.samples.append(tokenized)

        print(f"Loaded {len(self.samples)} samples")
        self.num_samples = len(self.samples)

    def __iter__(self):
        """Return iterator over batches"""
        indices = list(range(self.num_samples))
        random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.samples[idx] for idx in batch_indices]
            if len(batch) > 0:
                yield collate_multilabel(batch)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# ============ 5. REGISTER COMPONENTS ============

# Register in the original TUTA dictionaries
ptm.MODELS["multilabel"] = TUTAMultiLabelNative
tnr.TRAINERS["multilabel"] = train_multilabel
dymdata.DataLoaders["multilabel"] = MultiLabelDataLoader

print("✓ Multi-label components registered!")
print("Use: python train.py --target multilabel --dataset_paths your_data.json ...")
