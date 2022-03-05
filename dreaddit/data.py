import pandas as pd


def get_data():
    """method to get the data from CSVs"""
    training_data = pd.read_csv("../raw_data/dreaddit-train.csv")
    test_data = pd.read_csv("../raw_data/dreaddit-test.csv")
    #print('success')
    return training_data, test_data


def clean_data(df):
    cols_to_remove = [
        'syntax_ari', 'syntax_fk_grade', 'lex_dal_max_pleasantness',
        'lex_dal_max_activation', 'lex_dal_max_imagery',
        'lex_dal_min_pleasantness', 'lex_dal_min_activation',
        'lex_dal_min_imagery', 'lex_dal_avg_activation', 'lex_dal_avg_imagery',
        'lex_dal_avg_pleasantness', 'sentiment', 'post_id', 'social_timestamp',
        'subreddit', 'sentence_range'
    ]
    df.drop(columns=cols_to_remove, inplace=True)

    rows_to_remove = df['text'].loc[df['text'].apply(
        lambda x: len(x) < 35)].index.to_list()

    df.drop(labels=rows_to_remove, inplace=True)

    df['pct_caps'] = df['text'].apply(
        lambda x: sum([char.isupper() for char in x]) / len(x))
    df['text'] = df['text'].apply(lambda x: x.lower())

    df.set_index('id', inplace=True)

    df.drop(columns=['text'], inplace=True
            )  #keeping this separate if anything changes and we need the text

    #adding the capping on social variables (social_karma, 'social_num_comments',
    df['social_karma'] = df['social_karma'].apply(lambda x: 200
                                                  if x > 200 else x)
    df['social_num_comments'] = df['social_num_comments'].apply(
        lambda x: 100 if x > 100 else x)

    return df


if __name__ == '__main__':
    df = get_data()
