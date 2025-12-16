"""
Expected Completion Probability (ECP) Model
Trains a logistic regression model to predict pass completion probability
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

# Define Features (Matching your specific 'Supplementary' columns)
numeric_controls = [
    'pass_length',           # From Supplementary
    'yards_to_go',           # From Supplementary
    'score_differential',    # Calculated above
    'defenders_in_the_box',  # From Supplementary
    'passer_speed',          # From Snapshot
    'passer_acceleration',   # From Snapshot
    'passer_dist_to_sideline', # From Snapshot
    'separation_at_throw',   # From Snapshot
    'receiver_dist_to_sideline' # From Snapshot
]

categorical_controls = [
    'down',
    'quarter',
    'offense_formation',     # From Supplementary
    'receiver_alignment',    # From Supplementary
    'pass_location_type',    # From Supplementary
    'dropback_type',         # From Supplementary
    'team_coverage_man_zone' # From Supplementary (Man/Zone)
]

# Define Target
# 'pass_result' is 'C', 'I', 'IN', etc.
# y_binary = plays_df['pass_result'].apply(lambda x: 1 if x == 'C' else 0)

# Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_controls),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_controls)
    ])

baseline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='lbfgs', max_iter=2000))
])

# Train
# print("Training ECP Model on your data...")
# baseline_model.fit(plays_df, y_binary)

# Generate Target
# plays_df['ECP_Target'] = baseline_model.predict_proba(plays_df)[:, 1]
# print("ECP Targets Generated.")


def train_ecp_model(plays_df):
    """
    Train ECP model on plays dataframe
    
    Args:
        plays_df: DataFrame with play data including 'pass_result' column
    
    Returns:
        Trained model and target predictions
    """
    # Create binary target (1 = completion, 0 = incomplete)
    y_binary = plays_df['pass_result'].apply(lambda x: 1 if x == 'C' else 0)
    
    # Check which features are available
    available_numeric = [f for f in numeric_controls if f in plays_df.columns]
    available_categorical = [f for f in categorical_controls if f in plays_df.columns]
    
    print(f"Available numeric features: {len(available_numeric)}/{len(numeric_controls)}")
    print(f"Available categorical features: {len(available_categorical)}/{len(categorical_controls)}")
    
    if len(available_numeric) == 0 and len(available_categorical) == 0:
        raise ValueError("No features found in dataframe!")
    
    # Update preprocessor with available features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_numeric) if available_numeric else ('num', 'passthrough', []),
            ('cat', OneHotEncoder(handle_unknown='ignore'), available_categorical) if available_categorical else ('cat', 'passthrough', [])
        ],
        remainder='drop'
    )
    
    # Create model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=2000))
    ])
    
    # Train
    print("Training ECP Model on your data...")
    model.fit(plays_df, y_binary)
    
    # Generate predictions
    plays_df['ECP_Target'] = model.predict_proba(plays_df)[:, 1]
    print("ECP Targets Generated.")
    
    return model, plays_df


if __name__ == "__main__":
    print("""
    ECP Model - Expected Completion Probability
    
    To use:
    
    import pandas as pd
    from eliz import train_ecp_model
    
    # Load your data
    plays_df = pd.read_csv('your_plays_data.csv')
    
    # Train model and generate ECP targets
    model, plays_df_with_ecp = train_ecp_model(plays_df)
    
    # Now plays_df_with_ecp has 'ECP_Target' column with completion probabilities
    """)
