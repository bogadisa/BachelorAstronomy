import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import time

seed = 12345
np.random.seed(seed)

def open_df(filename):
    """
    Simply opens the new file. In order to see more of the data we make sure
    all columns display when the dataframe is printed
    """
    pd.set_option("display.max_columns", None)
    df = pd.read_csv(filename)
    return df

def add_death(df):
    """
    We dont want to deal with actual dates when we feed our data into the NN,
    therefore we restructure the DATE_DIED data into wether the person is dead
    or not in DEATH. Since only patients who died have a date, we can easily 
    convert the two. We also drop MEDICAL_UNIT, as this categorie is specific
    to where the patient was admitted in the hospital.
    """
    df = df.drop(columns= ["MEDICAL_UNIT"])

    df.insert(loc=len(df.columns),column='DEATH',value=0)
    df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] != "9999-99-99", 1)
    df["DEATH"] = df["DEATH"].where(df["DATE_DIED"] == "9999-99-99", 2)
    df = df.drop(columns=["DATE_DIED"])
    return df

def filter_df(df, filter):
    """
    Some columns need special handeling. This includes PREGNANT and CLASIFFICATION_FINAL.
    All men will have missing data, and a large amount of women also; we assume that all
    patients with no data (meaning a value of 97, 98 or 99) are not pregnant.
    CLASIFFICATION_FINAL has a value <=4 if a test is negative, so these patients are not
    of interest.
    """
    for feature in df.columns:
        if not(feature in filter):
            df.drop(df.loc[(df[feature] == 97) | (df[feature] == 99) | (df[feature] == 98)].index, inplace=True)
        elif feature == "CLASIFFICATION_FINAL":
            df.drop(df.loc[(df[feature] > 3)].index, inplace=True)
            df = df.drop(columns=[feature])
        elif feature == "PREGNANT":
            df["PREGNANT"].replace(97, 2) #we assume if unspecified --> not pregnant we replace with 2 = not pregnant, in dataset to avoid elminating samples of males as they contain 97-99
            df["PREGNANT"].replace(99, 2)
    return df

def convert_df_bool(df):
    """
    The data was given in mostly 1 (yes) or 2 (no) values. Here we convert them over to
    0 or 1. We will deal with AGE later, so it is filtered out. CLASIFFICATION_FINAL also
    needs special handeling for reasons meantioned previously.
    """
    for feature in df.columns:
        if feature != "AGE" and feature != "CLASIFFICATION_FINAL":
            df[feature] = (df[feature] == 1) #bool: True if == 2, false if not
            df[feature] = df[feature].astype(int) #convert bool to int: true = 1, false = 0
        elif feature != "AGE":
            df[feature] = (df[feature] <= 3)
            df[feature] = df[feature].astype(int)

        df[feature] = df[feature].astype(np.float32)
    return df

def create_age_groups(df, age_groups):
    """
    We want to have age in one-hot vector shape. To simplify and also strenghten
    correlation between old age and high risk, we first convert ages to age groups.
    The age groups range [0, 120]
    """
    for i in range(len(age_groups)-1):
        df.insert(loc=len(df.columns), column=f"AGE_GROUP_{i+1}",value=0)
        df[f"AGE_GROUP_{i+1}"] = df["AGE"].apply(lambda x: 1 if (x>age_groups[i] and x<age_groups[i+1]-1) else 0)

    df = df.drop(columns=["AGE"])
    return df

def define_target(df, target_name, target_par):
    """
    The data has several possible targets. This function allows us
    to easily collect several possible targets under one.
    """
    df[target_name] = 0
    for par in target_par:
        df[target_name] += df[par]
    
    code = f"df.{target_name} = df.{target_name}.apply(lambda x: 1 if x>0 else 0)"
    exec(code)
    df = df.drop(columns=target_par)
    return df

def balance_df(df, target_name):
    """
    Downsamples the dataset so that there are an equal amount of true targets as false.
    The majority is assumed to be True.
    """
    df_majority = df.loc[df[target_name] == 1]
    df_minority = df.loc[df[target_name] == 0]
    df_majority_downsampled = df_majority.sample(n=df[target_name].value_counts()[0], random_state=seed)
    
    df_upsampled = pd.concat([df_minority, df_majority_downsampled])

    return df_upsampled

def get_df(filename, n=None, filter=["AGE", "CLASIFFICATION_FINAL", "ICU", "INTUBED", "PREGNANT"],
            age_groups=[0, 18, 30, 40, 50, 65, 75, 85, 121], target_name="HIGH_RISK",
            target_par=["DEATH", "INTUBED", "ICU"], balance=True):
    """
    Applies all functions so that you can easily get the data in a seperate file.
    For experimentation purposes extra parameters are included, but to make it
    easier to use the program they all have default values.
    """
    df = open_df(filename)
    df = add_death(df)
    df = filter_df(df, filter)
    df = convert_df_bool(df)
    df = create_age_groups(df, age_groups)
    df = define_target(df, target_name, target_par)
    if n == None:
        n = len(df)
    if balance:
        df = balance_df(df, target_name)
    return df.sample(n=n)

if __name__ == "__main__":
    df = get_df("covid_data.csv")
    print(df)
    # run correlation matrix and plot
    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()