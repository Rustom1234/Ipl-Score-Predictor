import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import joblib

#check
# Load data
df = pd.read_csv('datasets/ipl_data_2016.csv')
df_temp = pd.read_csv('datasets/matches.csv')

# Merge city data into the main DataFrame
df_temp = df_temp[['match_id', 'city']]
df = pd.merge(df, df_temp, on='match_id')

# Drop unnecessary columns and rows with missing values
df = df.drop(['extras_type', 'player_dismissed', 'dismissal_kind', 'fielder'], axis=1)
df.dropna(inplace=True)

# Create unique identifier for each innings
df['unique_inning'] = df['match_id'].astype(str) + "_" + df['inning'].astype(str)

# Calculate cumulative stats for runs, wickets, and balls
df['current_runs'] = df.groupby(['unique_inning'])['total_runs'].cumsum()
df['wickets_left'] = 10 - df.groupby(['unique_inning'])['is_wicket'].cumsum()
df['balls_bowled'] = df.groupby(['unique_inning']).cumcount() + 1

# Calculate current run rate (ensuring non-zero balls to avoid division by zero)
df['current_run_rate'] = df['current_runs'] * 6 / df['balls_bowled']
df['balls_left'] = np.maximum(0, 120 - df['balls_bowled'])

# Calculate current partnership runs, resetting after each wicket in the inning
df['partnership_runs'] = df.groupby(['unique_inning', df['is_wicket'].cumsum()])['total_runs'].cumsum()

# Calculate Average Runs Per Wicket (AR/W), handle cases with no wickets
df['average_runs_per_wicket'] = df['current_runs'] / (10 - df['wickets_left'])
df['average_runs_per_wicket'] = df['average_runs_per_wicket'].replace([np.inf, -np.inf], 0)

# Calculate current bowler wickets in the game for each bowler
df['bowler_wickets'] = df.groupby(['match_id', 'bowler'])['is_wicket'].cumsum()

# Create target variable `runs_x` by calculating total runs at the end of each inning
runs_df = df.groupby(['match_id', 'inning'])['current_runs'].max().reset_index()
runs_df = runs_df.rename(columns={'current_runs': 'runs_x'})

# Merge `runs_x` back into the main DataFrame
df = pd.merge(df, runs_df, on=['match_id', 'inning'], how='left')

# Select final columns to use for model training
final_df = df[['batting_team', 'bowling_team', 'city',
               'current_runs', 'balls_left', 'wickets_left', 'current_run_rate',
               'partnership_runs', 'average_runs_per_wicket', 'bowler_wickets', 'runs_x']]

# Shuffle the data
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Splitting data
X = final_df.drop(columns=['runs_x'])
y = final_df['runs_x']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Setting up the pipeline
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', StandardScaler()),
    ('step3', XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=25, random_state=1))
])

# Training the pipeline
pipe.fit(X_train, y_train)

# Evaluating model performance
y_pred = pipe.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Save the trained pipeline
joblib.dump(pipe, "model_pipelines/ipl_pipeline.pkl")
print("Model saved as ipl_pipeline.pkl")