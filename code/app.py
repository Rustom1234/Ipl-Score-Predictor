import streamlit as st
import pandas as pd
import joblib  # Use this for loading the trained pipeline

# Load pre-trained pipeline (Assuming saved as "ipl_pipeline.pkl")
pipe = joblib.load("model_pipeline/ipl_pipeline.pkl")


# List of IPL teams
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Delhi Daredevils', 'Gujarat Lions', 
    'Gujarat Titans', 'Kings XI Punjab', 'Kolkata Knight Riders', 'Lucknow Super Giants', 
    'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals', 'Rising Pune Supergiant', 
    'Royal Challengers Bangalore', 'Royal Challengers Bengaluru', 'Sunrisers Hyderabad'
]

# List of cities
cities = [
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Chandigarh', 'Chennai', 
    'Delhi', 'Dharamsala', 'Dubai', 'Guwahati', 'Hyderabad', 'Indore', 'Jaipur', 
    'Kanpur', 'Kolkata', 'Lucknow', 'Mohali', 'Mumbai', 'Navi Mumbai', 'Pune', 
    'Rajkot', 'Sharjah', 'Unknown', 'Visakhapatnam'
]

# App title
st.title("IPL Cricket Score Predictor")

# Team and city selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

city = st.selectbox('Select the Match City', sorted(cities))

# Match details
col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Enter Current Score', min_value=0, value=0)

with col4:
    overs = st.number_input('Overs Completed (greater than 5)', min_value=5.0, max_value=20.0, value=5.0)

with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=0)

# Bowler wickets input
bowler_wickets = st.number_input('Number of Wickets by Current Bowler', min_value=0, value=0)

# Runs scored in the last 5 overs
last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=0, value=0)

# Prediction logic
if st.button('Predict Score'):
    # Calculations for derived features
    balls_left = int((20 - overs) * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    # Calculate average runs per wicket (AR/W), handling cases with no wickets
    if wickets == 0:
        avg_runs_per_wicket = 0
    else:
        avg_runs_per_wicket = current_score / wickets

    # Prepare input for the model
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_runs': [current_score],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'current_run_rate': [crr],
        'partnership_runs': [last_five],
        'average_runs_per_wicket': [avg_runs_per_wicket],
        'bowler_wickets': [bowler_wickets]
    })

    # Make prediction
    result = pipe.predict(input_df)

    # Display the predicted score
    st.header(f"Predicted Final Score: {int(result[0])}")