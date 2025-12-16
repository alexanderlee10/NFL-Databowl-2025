import pandas as pd

training_df = pd.read_csv("route_dominance_training_data.csv")
#drop unwanted features
#List of unwanted features
unwanted_features = [
    "frame_type",
    "time_to_ball",
    "def_time_to_ball",
    "time_advantage",
    "initial_leverage",
    "dominance_score",
    "route",
    "pass_result",
    "target_name",
    "is_complete",
    "offense_formation",
    "reciever_alignment",
    "coverage_type",
    "down",
    "yards_to_go",
    "pass_length",
    "frame_id"
]

training_df = training_df.drop(columns=unwanted_features)
training_df.rename(columns={"continuous_frame": "frame_id"}, inplace=True)

#save the cleaned training dataframe
training_df.to_csv("cleaned_training_data.csv", index=False)

#print the shape of the cleaned training dataframe
print(f"Cleaned training dataframe shape: {training_df.shape}")

#print the columns of the cleaned training dataframe
print(f"Cleaned training dataframe columns: {training_df.columns}")

#print the first 5 rows of the cleaned training dataframe