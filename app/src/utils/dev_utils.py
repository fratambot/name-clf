# Here we have all utility functions for development
import pandas as pd

# I/O
def load_raw_data(filepath):
    df = pd.read_csv(filepath)
    print("csv loaded into dataframe of shape: ", df.shape)
    return df

def drop_duplicates(df, subset):
    if len(subset) > 0:
        duplicates = df[df.duplicated(subset, keep=False)]
        if len(duplicates) > 0:
            print(
                f"Found these duplicates (\
                {len(duplicates)}): {duplicates} in {df.name}"
            )
            df.drop_duplicates(subset, inplace=True)
        else:
            print("No duplicates found")
    else:
        duplicates = df[df.duplicated(keep=False)]
        if len(duplicates) > 0:
            print(
                f"Found these duplicates (\
                {len(duplicates)}): {duplicates} in {df.name}"
            )
            df.drop_duplicates(inplace=True)
        else:
            print("No duplicates found")
    # assert there are no duplicates
    assert len(df[df.duplicated(keep=False)]) == 0
    return df

# plots
def model_performance_plot(history_or_cm):
    # It receive, e.g. history and/or confusion matrix and it
    # produces loss(epochs) plot and heatmaps which are saved
    # in /atrifacts/results

    return None
