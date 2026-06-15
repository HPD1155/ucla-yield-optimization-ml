import pandas as pd
import numpy as np
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data(path: str, delim='\t'):
    df = pd.read_csv(path, delimiter=delim, encoding='utf-16')
    return df


def clean_data(df):
    df = df.drop(columns=['Calculation1'], axis=1) # Drop index column
    school_list = set(df['School']) # Create unique list of possible schools


    for item in school_list: # Iterate through the school list
        has_val = False # Set up variable to check if school has values

        # Check if there are any values in the school
        for i in df[df['School'] == item]['All']:

            # If value found, break loop, go to next school
            if not pd.isna(i):
                has_val = True
                break

        # If school has no values, remove it
        if not has_val:
            df = df[df['School'] != item]

    return df



def create_parseable_df(df):
    columns = df.columns[:df.columns.get_loc('Count')] # Get Beginning constant columns
    rows = [] # Initialize new "Dataframe"
    school_list = set(df['School']) # Get unique list of schools

    # Iterate through schools
    for school in school_list:
        new_row = {} # Set up new row

        # Get every instance of said school in the DataFrame
        for i, item in df[df['School'] == school].iterrows():

            # Add default values
            for col in columns:
                new_row[col] = df[df['School'] == school][col].iloc[0] # Add all constant values based on masked original DataFrame
            
            # Create App, Adm, Enr rows
            for j in df.columns[df.columns.get_loc('Count') + 1:]:
                # Create rows for each count
                new_row[f'{item['Count']}_{j}'] = item[j] if not pd.isna(item[j]) else 0 # Convert row into column

        rows.append(new_row) # Add dictionary to "DataFrame"


    new_df = pd.DataFrame(rows) # Create true DataFrame by converting list of dicts to PD

    # Sort alphabetically by County
    new_df = new_df.sort_values(by='County/State/Country')

    return new_df    


# Compute each acceptance and yield rate    
def compute_funnels(df):
    # Split each column and turn into a set
    cols = list(df.columns[3:])
    for i in range(len(cols)):
        cols[i] = cols[i].split("_")[1]
    keys = set(cols) # Created set

    # Iterate through keys list
    for key in keys:
        df[f'Acceptance_{key}'] = np.where(df[f'App_{key}'] == 0, 0, df[f'Adm_{key}'] / df[f'App_{key}']) # Calculate Acceptance for each category
        df[f'Yield_{key}'] = np.where(df[f'Adm_{key}'] == 0, 0, df[f'Enr_{key}'] / df[f'Adm_{key}']) # Calculate Yield for each category

    return df    


if __name__ == "__main__":
    df = load_data("../data/GENDER_NONCALI_HS.csv")
    df = clean_data(df)
    print(df.head())
    new_df = create_parseable_df(df)
    print(new_df.head())
    new_df = compute_funnels(new_df)

    new_df.to_csv('../cleaned_data/parseable_NONCALI_gender_data.csv', index=False)
    print(new_df.head())
    print(new_df.shape)