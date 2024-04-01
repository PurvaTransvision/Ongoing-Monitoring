from flask import Flask,request,jsonify
from sqlalchemy import create_engine, exc
from sqlalchemy.pool import QueuePool
from sqlalchemy import text
# Import the necessary modules for error handling and logging
import traceback
import sys
import pandas as pd
import ast
import logging
import numpy as np
import json
import uuid
import re
from datetime import datetime, timedelta
from functools import lru_cache
import Levenshtein
from fuzzywuzzy import fuzz
import warnings
from difflib import SequenceMatcher
from flask_cors import CORS
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")


# Define your constants and configurations
MATCH_THRESHOLDS = {
    "Match": 90,
    "WeakMatch": 60,
    "Mismatch": 0,
    "Missing": 0
}

# Regular expression pattern to match a 4-digit year
pattern = r'\b\d{4}\b'


def process_matching(df_result, input_data, match_scores, column_name):
    logging.debug(f"Processing matching for {column_name}")
    data_list_n = df_result[column_name].apply(
        lambda x: [data.strip("[] '") for data in x.split(",")] if isinstance(x, str) else []
    )
    data_list_n = data_list_n.apply(lambda x: [data for data in x if data.strip()])  # Remove empty strings
    match_scores_list = []
    for data_list in data_list_n:
        if not data_list:
            match_scores_list.append(match_scores.get("Missing", 0))
        else:
            similarity_matrix = np.array([
                [calculate(input_item.lower(), data.lower()) for data in data_list]  # Fix the iteration here
                for input_item in input_data  # Fix the iteration here
            ])
            max_similarity_scores = np.max(similarity_matrix, axis=1)
            match_score = np.vectorize(calculate_match_score)(max_similarity_scores, match_scores)
            max_match_score = np.max(match_score)  # Get the maximum match score
            match_scores_list.append(max_match_score)
    return match_scores_list

# Function to extract years from request dates
def extract_year_from_request(date_list):
    logging.debug("Extracting years from request dates")
    years = []
    for date_str in date_list:
        if date_str:
            year_matches = re.findall(r'\b\d{4}\b', date_str)
            if year_matches:
                years.extend([int(match) for match in year_matches])
            else:
                years.append(None)  # Append None for missing or invalid dates
        else:
            years.append(None)  # Append None for missing dates
    return years

# Function to calculate match score and match type for a single entry based on dates
def date_matching(input_years, year_list, match_scores_date, threshold_years=2):

    if not year_list:
        return match_scores_date.get("Missing")
    year_list = [year for year in year_list if year is not None]  # Filter out None values
    if not input_years or not year_list:
        return match_scores_date.get("Missing")
    max_scores = []
    for request_year in input_years:
        if request_year is not None:
            for year_entry in year_list:
                if year_entry is not None:
                    difference_years = abs(year_entry - request_year)
                    match_score = match_scores_date.get("Match", 0) if difference_years == 0 else 0
                    match_score = match_scores_date.get("WeakMatch", 0) if 0 < difference_years <= threshold_years else match_score
                    match_score = match_scores_date.get("Mismatch", 0) if difference_years > threshold_years else match_score
                    max_scores.append(match_score)
    max_match_score = max(max_scores) if max_scores else 0
    return max_match_score


# Function to create a mapping of match scores to match types
def create_match_type_mapping(match_scores):
    logging.debug("Creating match type mapping")
    return {
        match_scores.get("Match", 0): "Match",
        match_scores.get("Mismatch", 0): "Mismatch",
        match_scores.get("WeakMatch", 0): "WeakMatch",
        match_scores.get("Missing", 0): "Missing"
    }


# Define a function for calculating match score
def calculate_similarity(input_name, names):
    if not input_name or input_name.strip() == "":
        logging.error("No party_name found in the request")
        return 0
    max_score = 0
    for name in names:
        name = name.strip()
        score = calculate(input_name.lower(), name.lower())
        max_score = max(max_score, score)
    return round(max_score, 2)

# Define a function for calculating match score for aircraft
def calculate_similarity_aircraft(input_name, names):
    if not input_name or input_name.strip() == "":
        logging.error("No party_name found in the request")
        return 0
    max_score = 0
    for name in names:  
        name = name.strip()
        score = calculate_aircraft(input_name.lower(), name.lower())
        max_score = max(max_score, score)
    # #print(max_score)
    return round(max_score, 2)

def clean_name(name):
    # Remove special characters, including single quotes, and extra spaces
    cleaned_name = re.sub(r"[^a-zA-Z\s]+", "", name).strip().lower()
    cleaned_name = ' '.join(cleaned_name.split())
    return cleaned_name

# def clean_name(name):
#     # Remove special characters, including single quotes, and extra spaces
#     cleaned_name = re.sub(r"[^a-zA-Z0-9\s-]+", "", name).strip().lower()
#     cleaned_name = ' '.join(cleaned_name.split())
#     return cleaned_name


def calculate_final_score_single(partyName, FullName):
    # Clean and format the input strings
    partyName = clean_name(partyName)
    FullName = clean_name(FullName)
    
    # #print(f"FullName - {FullName}")

    # Determine the larger and smaller names based on length
    if len(re.findall(r'\b\w+\b', partyName)) > len(re.findall(r'\b\w+\b', FullName)):
        larger_name, smaller_name = partyName, FullName
        
    # elif len(re.findall(r'\b\w+\b', partyName)) == len(re.findall(r'\b\w+\b', FullName)):
    #     larger_name, smaller_name = FullName, partyName
        
    elif len(re.findall(r'\b\w+\b', partyName)) == len(re.findall(r'\b\w+\b', FullName)):
    #         # Compare character lengths
        if len(FullName) > len(partyName):
            larger_name, smaller_name = FullName, partyName
            
        else:
            larger_name, smaller_name = partyName, FullName        
    

    elif len(re.findall(r'\b\w+\b', partyName)) < len(re.findall(r'\b\w+\b', FullName)):
        larger_name, smaller_name = FullName, partyName

    # Split the names into lists of words using regular expression
    partyWords = re.findall(r'\b\w+\b', smaller_name)
    words = re.findall(r'\b\w+\b', larger_name)
    
    party_length = len(re.findall(r'\b\w+\b', smaller_name))
    full_length = len(re.findall(r'\b\w+\b', larger_name))
    

    # Calculate total length excluding spaces
    total_length = len(''.join(larger_name.split()))

    # Calculate length per word dynamically
    lengths_per_word = [len(word) for word in words]

    # Calculate weightage for each word
    weightage_per_word = [(length / total_length) * 100 for length in lengths_per_word]

    # Store words, lengths, and weightage separately using lists
    word_list = words[:]
    lengths = lengths_per_word[:]
    weightages = weightage_per_word[:]

    # #print the results
    # #print("\nTotal Length:", total_length)
    # #print("Number of Words:", len(words))
    # #print("Words:", word_list)
    # #print("Lengths per Word:", lengths)
    # #print("Weightage per Word:", weightages)

    # Initialize the final score
    final_score = 0
    similarity_scores = []

    # Calculate similarity scores for each word in partyName and FullName
    # Calculate similarity scores for each word in partyName and FullName
    similarity_matrix = [[calculate(party_word, word) for word in words] for party_word in partyWords]

    # #print the similarity matrix
    # #print("\nSimilarity Matrix:")
    # for row in similarity_matrix:
        # #print(row)
        
    # #print the similarity matrix with indexes
    # #print("\nSimilarity Matrix:")
    # for i, row in enumerate(similarity_matrix):
    #     for j, score in enumerate(row):
            # #print(f"Index({i}, {j}): {score}")
            
            
    # Iterate through the similarity matrix
    for _ in range(len(similarity_matrix)):
        # Initialize variables to track maximum score and its indexes
        max_score = 0
        max_index = None

        # Iterate through the similarity matrix to find the maximum score and its indexes
        for i, row in enumerate(similarity_matrix):
            for j, score in enumerate(row):
                # Check if the current score is greater than the current maximum
                if score > max_score:
                    max_score = score
                    max_index = (i, j)

        # Check if a maximum score was found
        if max_index is not None:
            # #print the maximum score and its indexes
            # #print("\nMaximum Score:")
            # #print("Score:", max_score)
            # #print("Index:", max_index)

            # Pick the weight for the word corresponding to max_index from weightage_per_word
            word_weight = weightages[max_index[1]]
            # #print("Word Weight:", word_weight)

            
                # #print(f"Max Score after subtraction - {max_score}")

            # Multiply the max_score by the word_weight and add it to the final_score
            final_score += max_score * (word_weight / 100.0)
            
            # #print(f"Calculated Weight - {max_score * (word_weight / 100.0)}")

            # Remove the word from word_list corresponding to max_index
            removed_word = word_list.pop(max_index[1])
            # #print("Removed Word:", removed_word)
            
            removed_weight = weightages.pop(max_index[1])
            # #print("Removed Weight:", removed_weight)

            # Remove the row and column corresponding to the maximum score's index
            similarity_matrix.pop(max_index[0])
            for row in similarity_matrix:
                row.pop(max_index[1])

            # #print the updated similarity matrix
            # #print("\nUpdated Similarity Matrix:")
            # for row in similarity_matrix:
            #     #print(row)
        else:
            # If no maximum score is found, exit the loop
            break
    # # Check if the lengths are different and subtract the difference from max_score
    # if full_length != party_length:
    #     difference_in_length = abs(full_length - party_length)
    #     final_score -= 2.5 * difference_in_length

    # #print("\nFinal Score:", final_score)
    return round(final_score, 2)

def calculate_final_score(partyName, FullName):
    # If FullName is a list of names
    if isinstance(FullName, list):
        name_scores = []
        for name in FullName:
            # #print("\nProcessing Name:", name)
            name_score = calculate_final_score_single(partyName, name)
            name_scores.append(name_score)

        # Return the total score among all names
        if name_scores:
            return max(name_scores)
        else:
            # #print("\nNo name scores found.")
            return 0
    else:
        # FullName is a single name
        return calculate_final_score_single(partyName, FullName)

# Function to perform the KYC search for a single request
def kyc_search_single(df_request, df_sanction, sanction_list_name, config_match_scores):
    try:
        # Extracting party_type from df_request
        party_type = df_request['partyType'].iloc[0]

        logging.debug("Filtering party as per type")
        df_sanction.fillna("", inplace=True)

        def remove_empty_lists_and_none(value):
            if isinstance(value, list):
                if all(item == '' for item in value) or all(item is None for item in value):
                    return ''
            return value

        df_sanction = df_sanction.applymap(remove_empty_lists_and_none)

        # Remove empty strings from DataFrame
        def remove_empty_strings(value):
            if isinstance(value, list):
                return [item for item in value if item != '']
            else:
                return value

        df_sanction = df_sanction.applymap(remove_empty_strings)

        #Converting entityt columns to str types
        for col in df_sanction.columns:
            df_sanction[col] = df_sanction[col].astype(str)

        # Filtering df_master based on sdnType matching party_type
        filtered_df_master = df_sanction[df_sanction['sdnType'] == party_type]
        logging.debug("Filtered party df created!")

        # Extract the party_name from the data_request dictionary
        party_name = df_request['partyName'].iloc[0]

        # Split the party_name into words using whitespace as the delimiter
        words = party_name.split()

        # Count the number of words in the party_name
        num_words = len(words)

        # Convert AkaFullName column to a list of lists
        filtered_df_master['AkaFullName'] = filtered_df_master['AkaFullName'].apply(lambda x: ast.literal_eval(x) if x else []) # converting string representation into actual list


        filter_threshold = df_request['filter_threshold'].iloc[0]

        if filter_threshold is None or filter_threshold == 0:
            filter_threshold = 55

        #print(filter_threshold)

        if party_type == 'Aircraft':
            logging.debug("Processing similarity matching for FullName")

            # Calculate match score for FullName and store it in a new column
            filtered_df_master['fullName_match_score'] = filtered_df_master['FullName'].apply(
                lambda name: calculate_similarity_aircraft(party_name.lower(), [name.strip().lower()])
            )

            # Calculate match score for Party Name with AkaFullName and store it in a new column
            filtered_df_master['AkafullName_match_score'] = filtered_df_master['AkaFullName'].apply(
                lambda names: calculate_similarity_aircraft(party_name.lower(), [name.strip().lower() for name in names])
            )

            # Calculate the maximum match score between FullName and AkafullName
            filtered_df_master['nameMatchScore'] = filtered_df_master[['fullName_match_score', 'AkafullName_match_score']].max(axis=1)

            # Filter rows with a name_match_score greater than 70
            df_result = filtered_df_master[filtered_df_master['nameMatchScore'] >= int(filter_threshold)]

            df_result['AkaFullName'] = df_result['AkaFullName'].apply(str)
            num_records = len(df_result['FullName'])
            logging.debug(f"{num_records} records found with {party_type} as party_type")

            if len(df_result) == 0:
                logging.warning(f"No results found for given party_name in Akafullname as {party_type}")
                # #print(f"time for algorithm: {time.time()-kyc_time}")
                return df_result
        else:

            # Calculate match score for FullName and store it in a new column
            filtered_df_master['fullName_match_score'] = filtered_df_master['FullName'].apply(
                lambda name: calculate_final_score(party_name.lower(), [name.strip().lower()])
            )

            # Calculate match score for Party Name with AkaFullName and store it in a new column
            filtered_df_master['AkafullName_match_score'] = filtered_df_master['AkaFullName'].apply(
                lambda names: calculate_final_score(party_name.lower(), [name.strip().lower() for name in names])
            )

            # Calculate the maximum match score between FullName and AkafullName
            filtered_df_master['nameMatchScore'] = filtered_df_master[['fullName_match_score', 'AkafullName_match_score']].max(axis=1)

            # # Filter rows with a name_match_score greater than 70
            df_result = filtered_df_master[filtered_df_master['nameMatchScore'] >= int(filter_threshold)]

            # df_result = filtered_df_master

            df_result['AkaFullName'] = df_result['AkaFullName'].apply(str)
            # #print(df_result)
            num_records = len(df_result['FullName'])
            logging.debug(f"{num_records} records found with {party_type} as party_type")

        if len(df_result) == 0:
            logging.warning(f"No results found for given party_name in Akafullname as {party_type}")
            # #print(f"time for algorithm: {time.time()-kyc_time}")
            return df_result

    #     if num_words > 1:
    #         logging.debug("Processing similarity matching for FullName")

    #         # Calculate match score for FullName and store it in a new column
    #         filtered_df_master['fullName_match_score'] = filtered_df_master['FullName'].apply(
    #             lambda name: calculate_similarity(party_name.lower(), [name.strip().lower()])
    #         )

    #         # Calculate match score for Party Name with AkaFullName and store it in a new column
    #         filtered_df_master['AkafullName_match_score'] = filtered_df_master['AkaFullName'].apply(
    #             lambda names: calculate_similarity(party_name.lower(), [name.strip().lower() for name in names])
    #         )

    #         # Calculate the maximum match score between FullName and AkafullName
    #         filtered_df_master['nameMatchScore'] = filtered_df_master[['fullName_match_score', 'AkafullName_match_score']].max(axis=1)

    #         # Filter rows with a name_match_score greater than 70
    #         df_result = filtered_df_master[filtered_df_master['nameMatchScore'] > 70]

    #         df_result['AkaFullName'] = df_result['AkaFullName'].apply(str)
    #         num_records = len(df_result['FullName'])
    #         logging.debug(f"{num_records} records found with {party_type} as partyType")

    #         if len(df_result) == 0:
    #             logging.warning(f"No results found for given party_name in Akafullname as {party_type}")
    #             #grouped_df = df_request.groupby('record_id')
    #             empty_df = pd.DataFrame()
    #             return empty_df

    #     # PARTIAL MATCHING
    #     else:
    #         def tokenize_string(s):
    #             return re.findall(r'\w+', s.lower())  # This tokenization is more flexible
    #         # Create an empty list to store the weighted scores
    #         weighted_scores = []
    #         for party_name_token in df_request['partyName']:
    #             tokenized_party_name = tokenize_string(party_name_token)[0]
    #             for record in filtered_df_master['FullName']:
    #                 tokenized_record = tokenize_string(record)
    #                 # Split the tokenized_record into two halves
    #                 num_tokens = len(tokenized_record)
    #                 if num_tokens == 1:
    #                     first_half = tokenized_record[0]
    #                     last_half = ''
    #                 else:
    #                     half_length = num_tokens // 2
    #                     first_half = ' '.join(tokenized_record[:half_length])
    #                     last_half = ' '.join(tokenized_record[half_length:])
    #                 # Calculate scores for both halves
    #                 first_score = fuzz.ratio(tokenized_party_name, first_half)
    #                 last_score = fuzz.ratio(tokenized_party_name, last_half)
    #                 if first_score >= 83 or last_score >= 83:
    #                     weighted_score = (first_score * 0.6) + (last_score * 0.4)
    #                     # Append the weighted score to the list
    #                     weighted_scores.append(weighted_score)
    #                 else:
    #                     # Append a zero score if the condition is not met
    #                     weighted_scores.append(0)
    #         # Add a new column to the filtered_df_master with the weighted scores
    #         filtered_df_master['fullName_match_score'] = weighted_scores
    #         filtered_df_master['fullName_match_score'] = filtered_df_master['fullName_match_score'].round(2)
    #         # Filter the dataframe by the score threshold and store it in df_result

    #         #Partial Match for AkaFullname starts from here
    #         filtered_matching_records = []
    #         weighted_scores = [] # Added this list to store the weighted scores
    #         for party_name_token in df_request['partyName']:
    #             tokenized_party_name = tokenize_string(party_name_token)[0]
    #             matched_records = []  # To store the matched records
    #             for name_list in filtered_df_master['AkaFullName']:
    #                 max_first_score = 0
    #                 max_last_score = 0
    #                 matched_name = None
    #                 for record in name_list:
    #                     tokenized_record = tokenize_string(record)
    #                     num_tokens = len(tokenized_record)
    #                     if num_tokens == 1:
    #                         first_half = tokenized_record[0]
    #                         last_half = ''
    #                     else:
    #                         half_length = num_tokens // 2
    #                         first_half = ' '.join(tokenized_record[:half_length])
    #                         last_half = ' '.join(tokenized_record[half_length:])
    #                     first_score = fuzz.ratio(tokenized_party_name, first_half)
    #                     last_score = fuzz.ratio(tokenized_party_name, last_half)
    #                     max_first_score = max(max_first_score, first_score)
    #                     max_last_score = max(max_last_score, last_score)
    #                     if first_score >= 70 or last_score >= 70:
    #                         matched_name = record
    #                         break
    #                 if matched_name is not None:
    #                     weighted_score = round(max_first_score * 0.6 + max_last_score * 0.4, 2) # Added round function to store the score up to two decimals
    #                     matched_records.append({
    #                         'AkaFullName': name_list,
    #                         'MaxFirstScore': max_first_score,
    #                         'MaxLastScore': max_last_score,
    #                         'FullNameMatchScore': weighted_score # Used the rounded score here
    #                     })
    #                     weighted_scores.append(weighted_score) # Used the rounded score here
    #                 else:
    #                     weighted_scores.append(0) # Added this line to store zero score for each unmatched record
    #             if matched_records:
    #                 filtered_matching_records.extend(matched_records)

    #         filtered_df_master['AkafullName_match_score'] = weighted_scores

    #         filtered_df_master['nameMatchScore'] = filtered_df_master[['fullName_match_score', 'AkafullName_match_score']].max(axis=1)

    #         df_result = filtered_df_master[filtered_df_master['nameMatchScore'] > 55]

    #     df_result['AkaFullName'] = df_result['AkaFullName'].apply(str)
    #     df_result = df_result.drop_duplicates(subset=['uid', 'sdnType', 'Gender', 'idNumber', 'country', 'nationality',
    #    'dateOfBirth', 'FullName', 'AkaFullName', 'dateOfYear'])

    #     if len(df_result) == 0:
    #         logging.warning(f"No results found for given party_name in Akafullname as {party_type}")
    #         empty_df = pd.DataFrame()
    #         return empty_df

#           Alias Match Scores
        if 'partyAlias' in df_request.columns:
            input_alias_list = df_request['partyAlias']
            flattened_alias_list = [alias for sublist in input_alias_list for alias in sublist]
            flattened_alias_list = list(set(flattened_alias_list))

            match_scores_a = {}

            if not flattened_alias_list:
                df_result['aliasMatchScore'] = 0
                df_result['aliasMatchType'] = 'Missing'
            else:
                match_scores_a = config_match_scores.get("matchType", {}).get("partyAlias", {})

                alias_scores_list = []

                for input_alias in input_alias_list:

                    alias_list = process_matching(df_result, input_alias, match_scores_a, 'AkaFullName')
                    alias_scores_list.append(alias_list)

                max_alias_scores = [max(scores) for scores in zip(*alias_scores_list)]
                df_result['aliasMatchScore'] = max_alias_scores

            match_type_mapping_alias = create_match_type_mapping(match_scores_a)
            df_result['aliasMatchType'] = df_result['aliasMatchScore'].map(match_type_mapping_alias)
        else:
             # Handle missing 'partyAlias' column in the input DataFrame
            logging.warning("'partyAlias' column missing in input DataFrame")
            df_result['aliasMatchScore'] = 0
            df_result['aliasMatchType'] = 'Missing'



    #     Nationality Type Scores
        if 'idNationality' in df_request.columns:
            input_nation_list = df_request['idNationality']
            flattened_nation_list = [nation for sublist in input_nation_list for nation in sublist]
            flattened_nation_list = list(set(flattened_nation_list))
            match_scores_n = {}
            if not flattened_nation_list:
                df_result['nationalityMatchScore'] = 0
                df_result['nationalityMatchType'] = 'Missing'
            else:
                match_scores_n = config_match_scores.get("matchType", {}).get("idNationality", {})
                nation_scores_list = []
                for input_nation in input_nation_list:
                    nation_list = process_matching(df_result, input_nation, match_scores_n, 'nationality')
                    nation_scores_list.append(nation_list)
                max_nation_scores = [max(scores) for scores in zip(*nation_scores_list)]
                df_result['nationalityMatchScore'] = max_nation_scores

            match_type_mapping_nation = create_match_type_mapping(match_scores_n)
            df_result['nationalityMatchType'] = df_result['nationalityMatchScore'].map(match_type_mapping_nation)

        else:
             # Handle missing 'idNationality' column in the input DataFrame
            logging.warning("'idNationality' column missing in input DataFrame")
            df_result['nationalityMatchScore'] = 0
            df_result['nationalityMatchType'] = 'Missing'

        # Country Match Scores
        if 'countryOfResidence' in df_request.columns:
            # Country Match Scores
            input_country_list = df_request['countryOfResidence']
            flattened_country_list = [country for sublist in input_country_list for country in sublist]
            flattened_country_list = list(set(flattened_country_list))

            match_scores_c = {}
            if not flattened_country_list:
                df_result['countryMatchScore'] = 0
                df_result['countryMatchType'] = 'Missing'
            else:
                match_scores_c = config_match_scores.get("matchType", {}).get("countryOfResidence", {})
                country_scores_list = []
                for input_country in input_country_list:
                    country_list = process_matching(df_result, input_country, match_scores_c, 'country')
                    country_scores_list.append(country_list)
                max_country_scores = [max(scores) for scores in zip(*country_scores_list)]
                df_result['countryMatchScore'] = max_country_scores

            match_type_mapping_country = create_match_type_mapping(match_scores_c)
            df_result['countryMatchType'] = df_result['countryMatchScore'].map(match_type_mapping_country)
        else:
             # Handle missing 'countryOfResidence' column in the input DataFrame
            logging.warning("'countryOfResidence' column missing in input DataFrame")
            df_result['countryMatchScore'] = 0
            df_result['countryMatchType'] = 'Missing'


        # Party ID Match Scores
        if 'partyID' in df_request.columns:
            input_party_id = df_request['partyID'].iloc[0]
            match_scores_id = config_match_scores.get("matchType", {}).get("partyID", {})

            id_list = []
            logging.debug("Processing similarity matching for idNumber")
            logging.debug("Calculating matching score for idNumber")

            for _, row in df_result.iterrows():
                id_n_list_str = row['idNumber']

                if not id_n_list_str:  # Handle empty idNumber
                    match_score = match_scores_id.get("Missing", 0)
                    id_list.append(match_score)
                    continue

                # Convert the string representation of a list to an actual list
                id_number_list = eval(id_n_list_str) if isinstance(id_n_list_str, str) else [id_n_list_str]

                if len(id_number_list) == 0 or not any(input_party_id.values()):
                    match_score = match_scores_id.get("Missing", 0)
                    id_list.append(match_score)
                else:
                    id_sim = []

                    for input_id_key, input_id_value in input_party_id.items():
                        for id_number in id_number_list:
                            id_similarity = round(calculate(input_id_value, id_number), 2)
                            id_sim.append(id_similarity)

                    if len(id_sim) > 0:
                        max_similarity = max(id_sim)
                    else:
                        max_similarity = 0

                    match_score = calculate_match_score(max_similarity, match_scores_id)
                    id_list.append(match_score)

            df_result['partyIdMatchScore'] = id_list

            match_type_mapping_id = create_match_type_mapping(match_scores_id)

            df_result['partyIdMatchType'] = df_result['partyIdMatchScore'].map(match_type_mapping_id)
        else:
             # Handle missing 'party_id' column in the input DataFrame
            logging.warning("'party_id' column missing in input DataFrame")
            df_result['partyIdMatchScore'] = 0
            df_result['partyIdMatchType'] = 'Missing'


     # THIS IS FOR GENDER MATCHING

        # Gender Match Scores
        if 'partyGender' in df_request.columns:
            input_gender = df_request['partyGender'].iloc[0]
            match_scores_g = config_match_scores.get("matchType", {}).get("partyGender", {})
            logging.debug(f"Processing similarity matching for Gender")
            if not input_gender or input_gender.strip() == "":
                df_result['genderMatchScore'] = 0
                df_result['genderMatchType'] = 'Missing'
            elif df_result['Gender'].iloc[0] is None or df_result['Gender'].iloc[0].strip() == "":
                df_result['genderMatchScore'] = 0
                df_result['genderMatchType'] = 'Missing'
            else:

                # Function to calculate matching score based on direct comparison
                def calculate_gender(input_gender, gen):
                    if input_gender.lower() == gen.lower():
                        return 10
                    else:
                        return -5

                # Calculate matching score for each row in 'Gender' column
                gen_similarity = df_result['Gender'].apply(lambda gen: calculate_gender(input_gender.lower(), gen.lower()))
                df_result['genderMatchScore'] = gen_similarity

            logging.debug(f"Calculating matching score for Gender")

            match_type_mapping_gender = create_match_type_mapping(match_scores_g)
            df_result['genderMatchType'] = df_result['genderMatchScore'].map(match_type_mapping_gender)
        else:
            # Handle missing 'partyGender' column in the input DataFrame
            logging.warning("'partyGender' column missing in input DataFrame")
            df_result['genderMatchScore'] = 0
            df_result['genderMatchType'] = 'Missing'




    # Date of Birth (DOB) Match Scores
        if 'partyDOB' in df_request.columns:
            logging.debug(f"Processing similarity matching for dateOfBirth")
            # Extract years from 'partyDOB' in df_request
            df_request['year'] = df_request['partyDOB'].apply(extract_year_from_request)

            # Extract years from 'dateOfYear' in df_result
            df_result['date'] = df_result['dateOfYear'].apply(lambda x: [] if x == "" else [int(year) for year in re.findall(r'\d{4}', x)])
            logging.debug(f"Calculating matching score for dateOfBirth")

            # Get match scores configuration for DOB
            match_scores_date = config_match_scores.get("matchType", {}).get("partyDOB", {})

            # Calculate 'dobMatchScore' for each row in df_result
            df_result['dobMatchScore'] = [
                date_matching(df_request['year'].iloc[0], date_list, match_scores_date, threshold_years=2)
                if len(date_list) > 0
                else match_scores_date.get("Missing")
                for date_list in df_result['date']
            ]
            # Create match type mapping for DOB
            match_type_mapping_dob = create_match_type_mapping(match_scores_date)


            # Map 'dobMatchScore' to 'dobMatchType' using the match type mapping
            df_result['dobMatchType'] = df_result['dobMatchScore'].map(match_type_mapping_dob)
        else:
            # Handle missing 'partyDOB' column in the input DataFrame
            logging.warning("'partyDOB' column missing in input DataFrame")
            df_result['dobMatchScore'] = 0
            df_result['dobMatchType'] = 'Missing'



        # PARTIAL NAME MATCH SCORE
        df_result['partialNameMatchScore'] = 5

        df_result[['genderMatchScore', 'genderMatchType',
                   'partyIdMatchScore', 'partyIdMatchType',
                   'countryMatchScore', 'countryMatchType',
                    'nationalityMatchScore', 'nationalityMatchType',
                   'aliasMatchScore', 'aliasMatchType',
                   'dobMatchScore', 'dobMatchType']]

        columns_to_include = [
            'nameMatchScore',
            'genderMatchScore',
            'partyIdMatchScore',
            'countryMatchScore',
            'nationalityMatchScore',
            'aliasMatchScore',
            'dobMatchScore',
        ]
        df_result['total_match_score'] = df_result[columns_to_include].sum(axis=1)

        columns_to_iterate = [
            'genderMatchScore',
            'partyIdMatchScore',
            'countryMatchScore',
            'nationalityMatchScore',
            'aliasMatchScore',
            'dobMatchScore',
        ]

        score = 100
        for index, row in df_result.iterrows():
            count = 0
            final = 0

            for i in columns_to_iterate:

                if row[i] != 0:

                    count += 1
            final += score + 10 * count
            df_result.at[index, 'finalScore'] = round((df_result.at[index, 'total_match_score'] / final) * 100, 2)

        df_result['AkaFullName'] = df_result['AkaFullName'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
        df_result['country'] = df_result['country'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
        df_result['country'] = df_result['country'].apply(lambda x: list(set(x)))# Removing Duplicate Countries

        df_result['nationality'] = df_result['nationality'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
        df_result['nationality'] = df_result['nationality'].apply(lambda x: list(set(x)))# Removing Duplicate Nationality

        df_result['idNumber'] = df_result['idNumber'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])

        # Convert dateOfBirth values to actual lists
        df_result['dateOfBirth'] = df_result['dateOfBirth'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])

        df_result["partialNameMatchType"]='Full Name'

        df_result["sanctionListName"]= sanction_list_name

    #CLEANING DATAFRAME

        # Rename columns using the rename() method
        new_names = {
            'FullName':'matchName',
            'sdnType': 'matchType',
            'Gender' :'matchGender',
            'idNumber':'matchID',
            'country':'matchCountry',
            'nationality':'matchNationality',
            'AkaFullName': 'matchAlias',
            'dateOfBirth':'matchDOB',
            'uid':'matchKey'
        }

        df_result.rename(columns=new_names, inplace=True)

        # List of columns to drop from df_result
        columns_to_drop = ['date', 'total_match_score', 'normalized_total_match_score']

        # Filter the columns to drop based on the intersection with df_result columns
        columns_to_drop = [col for col in columns_to_drop if col in df_result.columns]

        # Drop the filtered columns from df_result
        df_result.drop(columns=columns_to_drop, inplace=True)

        ### ADD REQUEST RECORD TO FINAL DATAFRAME
        col_request = ['year']
        col_request = [col for col in col_request if col in df_request.columns]
        df_request.drop(columns=col_request, inplace=True) #DROP COLUMN YEAR

        #TO GET CORRECT NUMBER OF ROWS USE BROADCAST
        num_records = df_result.shape[0]
        df_broadcasted = pd.concat([df_request] * num_records, ignore_index=True, axis=0)

        # TO CONCAT BOTH DATAFRAMES
        df_final = pd.concat([df_result.reset_index(drop=True), df_broadcasted.reset_index(drop=True)], axis=1)

        return df_final
    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        return None

def strip_spaces(x):
    # Check if x is a list
    if isinstance(x, list):
        # Apply strip to each element of the list and return the result
        return [s.strip() if isinstance(s, str) else s for s in x]
    elif isinstance(x, str):
        # Apply strip to the single string and return the result
        return x.strip()
    else:
        # For other types (e.g., float), return them unchanged
        return x

def convert_to_compatible_format(date_str):
    # Convert the date string to 'YYYY-MM-DD' format
    converted_date = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
    return converted_date


# Function to fetch data from MySQL based on search_list and a specific generation_date
def fetch_delta_data_with_generation_date(search_list, newDate, previousDate):

    # Convert the date strings to compatible format
    newDate = convert_to_compatible_format(newDate)
    # #print(f"db new - {newDate}")
    previousDate = convert_to_compatible_format(previousDate)
    # #print(f"db old - {previousDate}")

    #target_generation_date = "28-09-2023"
    query = f"""
        SELECT *
            FROM {search_list}
            WHERE TS_REFRESH_DATE >= '{previousDate} 00:00:00' AND TS_REFRESH_DATE < '{newDate} 23:59:59'


    """
    # Create a connection
    with engine.connect() as connection:
        # Execute the query using the connection and fetch data
        result = connection.execute(query)
        # Convert the result into a DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    
    # df = pd.read_sql(query, con=engine)
    return df

def calculate_match_score(similarity, match_scores):
    for match_type, threshold in MATCH_THRESHOLDS.items():
        if similarity >= threshold:
            return match_scores.get(match_type, 0)
    return match_scores.get("Mismatch", 0)



@lru_cache(maxsize=50000)
def calculate(str1, str2):
    # Clean and format the input strings
    # str1 = re.sub(r"[^a-zA-Z\s]+", " ", str1).strip().lower()
    # str2 = re.sub(r"[^a-zA-Z\s]+", " ", str2).strip().lower()

    str1 = re.sub(r"[^a-zA-Z0-9\s-]+", " ", str1).strip().lower()
    str2 = re.sub(r"[^a-zA-Z0-9\s-]+", " ", str2).strip().lower()

    # Remove extra spaces between names
    str1 = ' '.join(str1.split())
    str2 = ' '.join(str2.split())

    distance = Levenshtein.distance(str1, str2)

    # Calculate similarity as a percentage
    max_len = max(len(str1), len(str2))
    similarity = ((max_len - distance) / max_len) * 100

    # Calculate similarity between the cleaned strings
    # similarity = SequenceMatcher(None, str1, str2).ratio()

    return round(similarity, 2)


@lru_cache(maxsize=10000)
def calculate_aircraft(str1, str2):
    # Clean and format the input strings, excluding hyphens and numbers
    str1 = re.sub(r"[^a-zA-Z0-9\s-]+", " ", str1).strip().lower()
    str2 = re.sub(r"[^a-zA-Z0-9\s-]+", " ", str2).strip().lower()


    # Remove extra spaces between names
    str1 = ' '.join(str1.split())
    str2 = ' '.join(str2.split())

    distance = Levenshtein.distance(str1, str2)

    # Calculate similarity as a percentage
    max_len = max(len(str1), len(str2))
    similarity = ((max_len - distance) / max_len) * 100
    # #print(str1)
    # #print(str2)
    # Calculate similarity between the cleaned strings
    # similarity = SequenceMatcher(None, str1, str2).ratio()
    return round(similarity, 2)

def validate_client(client_id):
    try:
        # #print(client_id)
        engine = create_engine(db_connection_string)
        with engine.connect() as connection:
            # Fetch STATUS and EXPIRY_DATE from TS_ACCT table based on the given client_id
            # acct_info_query = f'SELECT STATUS, EXPIRY_DT FROM TS_ACCT WHERE CLIENT_ID = {int(client_id)}'
             # Create a text object from the SQL string
            acct_info_query = text('SELECT STATUS, EXPIRY_DT FROM TS_ACCT WHERE CLIENT_ID = :client_id')

            # Bind parameters
            acct_info_query = acct_info_query.bindparams(client_id=client_id)
            acct_info_result = connection.execute(acct_info_query)
            acct_info = acct_info_result.fetchone()

            if acct_info is not None:
                status, expiry_date = acct_info

                # Convert expiry_date to datetime for proper comparison
                # expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d %H:%M:%S')

                # Check if the status is "ACTIVE" and EXPIRY_DATE is greater than the current date
                current_date = datetime.now()
                if status == "ACTIVE" and expiry_date > current_date:
                    return True
                else:
                    return False
            else:
                #print(f"No matching entry found for CLIENT_ID: {client_id}")
                return False

    except exc.SQLAlchemyError as e:
        logging.error(f"SQLAlchemyError: {e}")
        # Handle the error or re-raise it as needed
        raise
    except Exception as e:
        logging.error(f"Error validating client: {e}")
        # Handle the error or re-raise it as needed
        raise
    finally:
        try:
            if engine:
                engine.dispose()
        except Exception as e:
            logging.error(f"Error closing the database connection: {e}")

def update_kyc_stats(client_id, request_count):

    try:
        engine = create_engine(db_connection_string)
        with engine.connect() as connection:
            # Fetch ACCT_ID from TS_ACCT_CLIENT table based on the given client_id
            acct_id_query = f"SELECT ACCT_ID FROM TS_ACCT_CLIENT WHERE CLIENT_ID = {client_id}"
            acct_id_result = connection.execute(acct_id_query)
            acct_id = acct_id_result.scalar()

            if acct_id is not None:
                # Update KYC count in TS_KYC_STATS table
                # Insert a new row into TS_KYC_STATS table
                insert_kyc_query = f"""
                    INSERT INTO TS_KYC_STATS (ACCT_ID, KYC, PEP, NNS, CREATE_DT)
                    VALUES ({acct_id}, {request_count}, 0, 0, GETDATE())
                """
                connection.execute(insert_kyc_query)
            else:
                logging.info(f"No matching ACCT_ID found for CLIENT_ID: {client_id}")
                #print(f"No matching ACCT_ID found for CLIENT_ID: {client_id}")

    except exc.SQLAlchemyError as e:
        logging.error(f"SQLAlchemyError: {e}")
        # Handle the error or re-raise it as needed
        raise
    except Exception as e:
        logging.error(f"Error updating KYC stats: {e}")
        # Handle the error or re-raise it as needed
        raise
    finally:
        try:
            if engine:
                engine.dispose()
        except Exception as e:
            logging.error(f"Error closing the database connection: {e}")


# Function to fetch data from MySQL based on search_list
def fetch_sanction_data(search_list):
    logging.debug(f"Loading {search_list} from Database")
    query = f"""
        SELECT *
        FROM {search_list};
    """
    # df = pd.read_sql(query, con=engine)
    # Create a connection
    with engine.connect() as connection:
        # Execute the query using the connection and fetch data
        result = connection.execute(query)
        # Convert the result into a DataFrame
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

config_file_path = "config.json"

def load_match_scores(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


config_match_scores = load_match_scores(config_file_path)

def log_error_and_return_response(error_msg):
    logging.error(error_msg)
    return jsonify({"error": error_msg})

# Initialize previous_data and new_data as None initially
previous_data = None
new_data = None

# Define global variables for previous_df and new_df
previous_df = None
new_df = None

# OUTPUT_FILE = 'ofac_output_client.json'

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='ongoing_monitoring_log.log',
                    filemode='w')

# Database connection string
db_connection_string =os.getenv("TS_SANCTION_CONNECTION")

engine = create_engine(db_connection_string, poolclass=QueuePool)

app = Flask(__name__)

@app.route('/api/monitor_search', methods=['POST'])
def change_monitoring():
    try:
        data = request.json
        # client_id = data.get("clientID")
        filter_threshold = data.get("filterThreshold")
        # request_count = data.get("requestCount")
        # client_status = validate_client(client_id)
        logging.info("Received a POST request to /api/monitor_search")

        # if not client_status:
        #     logging.error("No party_type found in the reqsuest")
        #     result_json = {
        #             "httpStatus": "BAD_REQUEST",
        #             "message": "Client is not Active or License Expired",
        #             "result": []
        #         }
        #     return result_json
        records = data.get("records", [])
        results = []

        #stores all the data for all records after searching through all lists
        all_results = []
        # Load and process each sanction list one by one
        record_result = []
        for record in records:
            logging.info("Processing a new record...")
            df_request = pd.DataFrame(record['fields'])
            ##print(df_request)
            df_request['record_id'] = record['id']
            df_request['filter_threshold'] = filter_threshold
            # 'record_id' is currently a string in df_request
            #df_request['record_id'] = df_request['record_id'].astype(int)
            record_id = int(df_request['record_id'].iloc[0])

            ##print(record_id)
            ##print(type(record_id))
            # List of fields and their corresponding keys
            required_fields = [
                ("partyName", "Party Name"),
                ("partyType", "Party Type")
            ]

            # Initialize a dictionary to store field values
            field_values = {}

            # Loop through the required fields
            for field_key, field_name in required_fields:
                try:
                    field_values[field_key] = df_request.get(field_key)
                    if field_values[field_key] is None:
                        raise KeyError  # Trigger KeyError for missing field
                except KeyError:
                    logging.error(f"Bad Request: {field_name} not found in the reqsuest")
                    error_msg = {
                    "httpStatus": "BAD_REQUEST",
                    "message": f"{field_name} is missing in the request",
                    "result": []
                    }

                    return log_error_and_return_response(error_msg)

            # Now you can safely access the field values
            party_name = field_values.get("partyName", "")
            partyType = field_values.get("partyType", "")

            party_type = df_request['partyType'].iloc[0]
            party_name = df_request['partyName'].iloc[0]



            # #print(df_request['previousDate'].iloc[0])
            # #print(df_request['newDate'].iloc[0])
            # check if the previous and new dates are empty consider T-1 dates for delta
            if (not df_request['previousDate'].iloc[0]) or (not df_request['newDate'].iloc[0]):
                # #print("I m here")
                # Get today's date
                newDate = datetime.now()
                # #print(newDate)
                # Subtract one day
                previousDate = newDate - timedelta(days=1)
                # #print(previousDate)
                # #print(newDate)
                # #print(previousDate)

            else:
                previousDate = df_request['previousDate'].iloc[0]
                newDate = df_request['newDate'].iloc[0]

            if not party_type or party_type.strip() == "":
                logging.error("No partyType found in the reqsuest")
                result_json = {
                        "httpStatus": "BAD_REQUEST",
                        "message": "Party type is missing in the request",
                        "result": []
                    }
                return result_json
            elif not party_name or party_name.strip() == "":
                logging.error("No partyName found in the request")
                result_json = {
                        "httpStatus": "BAD_REQUEST",
                        "message": "Party name is missing in the request",
                        "result": []
                    }
                return result_json


            # Create a dictionary to map each list to its corresponding delta list
            list_to_delta_mapping = {
                "TS_CANTELIST": "TS_CANTEDELTA",
                "TS_OFACLIST": "TS_OFACDELTA",
                "TS_UNLIST": "TS_UNDELTA",
                "TS_EULIST": "TS_EUDELTA",
                "TS_CANADALIST": "TS_CANADADELTA",
                "TS_WBLIST": "TS_WBDELTA",
                "TS_BIS_CSL_LIST": "TS_BIS_CSL_DELTA",
                "TS_UKLIST" : "TS_UKDELTA",
                "TS_AUSLIST" : "TS_AUSDELTA",
                "TS_OFACADVANCEDLIST": "TS_OFACADVANCEDDELTA",
                "TS_BELGIANLIST": "TS_BELGIANDELTA",
                "TS_SWISSLIST": "TS_SWISSDELTA",
                "TS_AZ_FMS_LIST": "TS_AZ_FMS_DELTA",
                "TS_UKRAINE_LIST": "TS_UKRAINE_DELTA",
                "TS_AZ_2231_LIST": "TS_AZ_2231_DELTA",
                "TS_NON_OFAC_LIST": "TS_NON_OFAC_DELTA",
                "TS_IADBLIST":"TS_IADBDELTA",
                "TS_EBRDLIST":"TS_EBRDDELTA",
                "TS_UKRAINELIST":"TS_UKRAINEDELTA"

            }


            # define all the list you want to iterate through
            listprefixes = ["TS_OFACLIST", "TS_CANADALIST", "TS_CANTELIST", "TS_UKLIST", "TS_UNLIST", "TS_EULIST", "TS_WBLIST", "TS_AUSLIST", "TS_BIS_CSL_LIST", "TS_OFACADVANCEDLIST", "TS_SWISSLIST","TS_IADBLIST","TS_EBRDLIST","TS_UKRAINELIST"]
            # listprefixes = ["TS_OFACLIST","TS_UNLIST","TS_EULIST","TS_CANADALIST","TS_CANTELIST","TS_WBLIST"]

            def kyc_search_single_for_table(searchlist):
                logging.info(f"KYC single search process started for {searchlist}")
                try:
                    # Check if the searchlist is in the mapping
                    if searchlist in list_to_delta_mapping:
                        delta_list = list_to_delta_mapping[searchlist]
                        try:
                            #
                            print("here")
                            df_sanction = fetch_delta_data_with_generation_date(delta_list, newDate, previousDate)
                            columns_to_keep = ['uid', 'sdnType', 'Gender', 'idNumber', 'country', 'nationality', 'dateOfBirth', 'FullName', 'AkaFullName', 'dateOfYear', 'Status', 'updated_values','Severity',
                                               'personalDetails','addressDetails','contactDetails','identifierDetails','regulatoryDetails','vesselDetails','carrierDetails','otherDetails']
                            df_sanction = df_sanction[columns_to_keep]
                            print(df_sanction)
                        except Exception as e:
                            logging.exception("An exception occurred:", str(e))


                    df_final = kyc_search_single(df_request, df_sanction, searchlist, config_match_scores)
                    return df_final

                except KeyError as e:

                    # #print("I m here")
                    error_field = e.args[0]
                    logging.error(f"KeyError - Field: {error_field}, Error: {str(e)}")

                    result_json = {
                        "httpStatus": "INTERNAL_SERVER_ERROR",
                        "error": "An error occurred during processing",
                        "result": ""
                    }

                    return result_json


            # Create a list to hold results for the current record
            record_results = {
                "httpStatus": "OK",
                # "message": "Record Found Successfully",
                "id": record_id,
                "result": {
                    "added": [],
                    "removed": [],
                    "updated": [],
                }
            }

            for prefix in listprefixes:
                df_result  = kyc_search_single_for_table(prefix)
                # #print(df_result)
                added_list = []
                removed_list = []
                updated_list = []
                if not df_result.empty:
                    # Group the DataFrame by the 'Record_ID' column
                    grouped_data = df_result.groupby('record_id')

                    # Initialize an empty list to store the matched dictionaries
                    matched_list = []

                    # Define the desired order of keys within each "matched" dictionary
                    desired_order = [
                        'finalScore','partyKey', 'matchKey',
                        'matchType', 'partyType', 'partyName', 'matchName', 'nameMatchScore',
                        'partyGender', 'matchGender', 'genderMatchScore', 'genderMatchType',
                        'partyID', 'matchID', 'partyIdMatchScore', 'partyIdMatchType',
                        'idNationality', 'matchNationality', 'nationalityMatchScore', 'nationalityMatchType',
                        'matchCountry', 'countryOfResidence', 'countryMatchScore', 'countryMatchType',
                        'partyAlias', 'matchAlias', 'aliasMatchScore', 'aliasMatchType',
                        'partyDOB', 'dateOfBirth', 'matchDOB', 'dobMatchScore', 'dobMatchType',
                        'partialNameMatchScore', 'partialNameMatchType', 'Status', 'updated_values','Severity','personalDetails','addressDetails', 'contactDetails','identifierDetails','regulatoryDetails','vesselDetails','carrierDetails','otherDetails'
                    ]

                    for record_id, group_df in grouped_data:
                        added_entries = []
                        removed_entries = []
                        updated_entries = []
                        for matched_entry in group_df.to_dict(orient='records'):
                            new_matched_entry = {key: matched_entry[key] for key in desired_order if key in matched_entry}
                            # Convert string representations of dictionaries to actual dictionaries
                            new_matched_entry['personalDetails'] = ast.literal_eval(new_matched_entry['personalDetails'])
                            new_matched_entry['addressDetails'] = ast.literal_eval(new_matched_entry['addressDetails'])
                            new_matched_entry['contactDetails'] = ast.literal_eval(new_matched_entry['contactDetails'])
                            new_matched_entry['identifierDetails'] = ast.literal_eval(new_matched_entry['identifierDetails'])
                            new_matched_entry['regulatoryDetails'] = ast.literal_eval(new_matched_entry['regulatoryDetails'])
                            new_matched_entry['vesselDetails'] =ast.literal_eval(new_matched_entry['vesselDetails'])
                            new_matched_entry['carrierDetails'] = ast.literal_eval(new_matched_entry['carrierDetails'])
                            new_matched_entry['otherDetails'] = ast.literal_eval(new_matched_entry['otherDetails'])
                            
                            # #print(new_matched_entry)
                            status = matched_entry['Status']

                            if status == "Added":
                                new_matched_entry['updated_values'] = []
                                added_entries.append(new_matched_entry)

                            elif status == "Removed":
                                new_matched_entry['updated_values'] = []
                                removed_entries.append(new_matched_entry)
                            elif status.startswith("Update"):
                                new_matched_entry['updated_values'] = ast.literal_eval(new_matched_entry['updated_values'])
                                # updated_entries['Updated Values'] = new_matched_entry['updated_values'']
                                updated_entries.append(new_matched_entry)
                            # if status in ["Update (minor changes)", "Update (major changes)"]:

                        added_list.append({
                            "sanctionList": prefix,
                            "matchCount": len(added_entries),
                            "matched": added_entries
                        })
                        removed_list.append({
                            "sanctionList": prefix,
                            "matchCount": len(removed_entries),
                            "matched": removed_entries
                        })
                        updated_list.append({
                            "sanctionList": prefix,
                            "matchCount": len(updated_entries),
                            "matched": updated_entries
                        })

                        # Append to the record_results
                        record_results["result"]["added"].extend(added_list)
                        record_results["result"]["removed"].extend(removed_list)
                        record_results["result"]["updated"].extend(updated_list)
                        # record_results["message"] = "Record Found Successfully"


                elif df_result.empty:
                    added_list.append({
                            "sanctionList": prefix,
                            "matchCount": 0,
                            "matched": []
                        })
                    removed_list.append({
                        "sanctionList": prefix,
                        "matchCount": 0,
                        "matched": []
                    })
                    updated_list.append({
                        "sanctionList": prefix,
                        "matchCount": 0,
                        "matched": []
                    })
                    # Append to the record_results
                    record_results["result"]["added"].extend(added_list)
                    record_results["result"]["removed"].extend(removed_list)
                    record_results["result"]["updated"].extend(updated_list)
                    # record_results["message"] = "No Record Found!"

                ##print(prefix)
                ##print(df_result)

            ##print(record_results)
            record_result.append(record_results)

        
        

        all_results.extend(record_result)
        # Loop over each result in all_results
        for record_results in all_results:
            # Initialize a flag to check if any matchCount is greater than 0
            record_found = False
            
            # Loop over each type of result (added, removed, updated)
            for result_type in ["added", "removed", "updated"]:
                # Loop over each entry in the current result type
                for entry in record_results["result"][result_type]:
                    # Check if matchCount is greater than 0
                    if entry["matchCount"] > 0:
                        # If any matchCount is greater than 0, set the flag to True and break the loop
                        record_found = True
                        break
                # If record_found is True, break the outer loop as well
                if record_found:
                    break
            
            # Check the value of record_found and update the message accordingly
            if record_found:
                record_results["message"] = "Record Found Successfully"
            else:
                record_results["message"] = "No Record Found!"
        logging.info("All records have been processed successfully.")

    except Exception as e:
        # #print("I m inside single")
        # Log any unhandled exceptions
        logging.error(f"An error occurred: {str(e)}")
        result_json = {
            "httpStatus": "INTERNAL_SERVER_ERROR",
            "error": "An error occurred during processing",
            "result": ""
        }
        return result_json
    


    # OUTPUT_FILE = "response.json"

    # with open(OUTPUT_FILE, 'w') as file:
    #     json.dump(all_results, file, indent=4)
    #     file.write('\n')

    return jsonify(all_results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6860, debug = False)