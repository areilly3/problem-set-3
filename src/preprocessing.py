'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Reads CSV's into df's
    model_pred_df = pd.read_csv('./data/prediction_model_03.csv')
    genres_df = pd.read_csv('./data/genres.csv')

    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    # Creates genre_list 
    genre_list = genres_df['genre'].dropna().unique().tolist()
    
    # Creates genre_true_counts
    genre_true_counts = {}
    
    # Gets counts of how many times each genre appears in actual_genres
    for actual_genres in model_pred_df['actual genres']:
        genres_list = ast.literal_eval(actual_genres) # Turns str(actual_genres) into list(genres_list)
        for genre in genres_list:
            genre_true_counts[genre] = genre_true_counts.get(genre, 0) + 1
    
    # Dictionary comprehension to make fn calulation easier later
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    
    # Gets counts of true postivies and false positives 
    for index, row in model_pred_df.iterrows():
        predicted_genre = row['predicted']  
        actual_genres = row['actual genres']      
        
        if predicted_genre in actual_genres:
            genre_tp_counts[predicted_genre] += 1
        else:
            genre_fp_counts[predicted_genre] += 1 

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
        