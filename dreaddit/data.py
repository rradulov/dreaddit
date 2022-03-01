import pandas as pd

def get_data():
    """method to get the data from CSVs"""

    training_data = pd.read_csv("../raw_data/dreaddit-train.csv")
    test_data = pd.read_csv("../raw_data/dreaddit-test.csv")

    #print('success')

    return training_data, test_data

def clean_data(df):
    cols_to_remove = ['syntax_ari',
                   'syntax_fk_grade',
                   'lex_dal_max_pleasantness',
                   'lex_dal_max_activation',
                   'lex_dal_max_imagery',
                   'lex_dal_min_pleasantness',
                   'lex_dal_min_activation',
                   'lex_dal_min_imagery',
                   'lex_dal_avg_activation',
                   'lex_dal_avg_imagery',
                   'lex_dal_avg_pleasantness',
                   'sentiment',
                   'post_id',
                   'social_timestamp']
    df = df.drop(columns=cols_to_remove, inplace=True)
    
    rows_to_remove = df['text'].apply(lambda x: len(x)<35)
    df.drop(labels=rows_to_remove, inplace=True)
    
    return df

if __name__ == '__main__':
    df = get_data()
    

