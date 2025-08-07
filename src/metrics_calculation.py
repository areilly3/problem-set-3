'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''
    # Gets counts for false negatives
    genre_fn = {}
    for genre in genre_list:
        genre_fn[genre] = genre_true_counts[genre] - genre_tp_counts[genre]
    
    # Sum of tp, fp, and fn values
    tp_sum = sum(genre_tp_counts.values())
    fp_sum = sum(genre_fp_counts.values())
    fn_sum = sum(genre_fn.values())
    
    # Equations for precision, recall, f1-score
    micro_precision = tp_sum / (tp_sum + fp_sum)
    micro_recall = tp_sum / (tp_sum + fn_sum)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    
    micro_metrics = (micro_precision, micro_recall, micro_f1)
    
    macro_precisions = []
    macro_recalls = []
    macro_f1_scores = []
    
    # Gets macro recision, recall, and f1-scores
    for genre in genre_list:
        macro_tp = genre_tp_counts[genre]
        macro_fp = genre_fp_counts[genre]
        macro_fn = genre_fp_counts[genre]
        
        macro_precision = macro_tp / (macro_tp + macro_fp) if (macro_tp + macro_fp) > 0 else 0
        macro_recall = macro_tp / (macro_tp + macro_fn) if (macro_tp + macro_fn) > 0 else 0 
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0 

        macro_precisions.append(macro_precision)
        macro_recalls.append(macro_recall)
        macro_f1_scores.append(macro_f1)
        
    macro_metrics = [macro_precisions, macro_recalls, macro_f1_scores]

    return micro_metrics, macro_metrics
        
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    pred_rows = []
    true_rows = []
    
    for index, row in model_pred_df.iterrows():
        predicted_genre = row['predicted']
        actual_genres = ast.literal_eval(row['actual genres'])
        
        pred_row = {}
        true_row = {}
        
        for genre in genre_list:
            pred_row[genre] = 1 if genre == predicted_genre else 0
            true_row[genre] = 1 if genre in actual_genres else 0
            
        pred_rows.append(pred_row)
        true_rows.append(true_row)
    
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    
    macro_precision, macro_recall, macro_f1, n = precision_recall_fscore_support(true_matrix, pred_matrix, average= 'macro', zero_division= 0.0)
    micro_precision, micro_recall, micro_f1, n = precision_recall_fscore_support(true_matrix, pred_matrix, average= 'micro', zero_division= 0.0)
    
    return (macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)