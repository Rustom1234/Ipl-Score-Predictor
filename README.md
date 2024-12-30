# IPL Cricket Score Predictor  

<img src="Screenshot 2024-12-30 at 10.25.29 AM.png" alt="Score Predictor App" width="400"/>

## Introduction

The IPL Cricket Score Predictor is a machine learning application designed to predict the final score of an Indian Premier League (IPL) cricket match based on various match parameters. Leveraging historical match data, the application trains a regression model to forecast the total runs a batting team is expected to score by the end of their innings. The downloaded dataset `ipl_data_2016.csv` gives a ball by ball  breakdown of a set of IPL games from 2016 onwards, accessed from Kaggle.

## Installation

1. **Clone the Repository**
```
git clone https://github.com/yourusername/ipl-score-predictor.git
cd ipl-score-predictor
```
2. **Install Dependencies**
```
pip install -r requirements.txt
```

## Datasets

1. **Dataset Files**
- `ipl_data_2016.csv`: Contains a ball-by-ball data of IPL 2016 matches.
- `matches.csv`: Contains summary details of IPL matches, including match IDs and cities.
2. **Ensure Data Availability:** Place both `ipl_data_2016.csv` and `matches.csv` in the project directory.
3. **Accessing the datafiles from orignal source**
- `ipl_data_2016.csv`: Downloaded from Kaggle at https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020?select=deliveries.csv. This gets downloaded as the file `deliveries.csv`, and has data from 2008 to 2024. Next manually delete rows of data from 2008 to 2015 and save the updated dataset as `ipl_data_2016.csv`.
- `matches.csv`: Download from Kaggle at https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020?select=matches.csv.

## Model Training

**Note:** This is not essential to using the app.

1. **Run the Training Script:** Execute the Python script to preprocess data, train the model, evaluate performance, and save the trained pipeline.
```
python model_pipeline.py
```
2. **Output**
- Model Metrics: The script will print the R<sup>2</sup> score and Mean Absolute Error.
- Saved Model: The trained pipeline is saved as `ipl_pipeline.pkl` in the project directory.

## Running the Streamlit App

1. **Launch the App**
```
streamlit run app.py
```
2. **Access the App:** Open your web browser and navigate to http://localhost:8501 to interact with the IPL Score Predictor if it does not open automatically.

## Usage
1. **Input Match Details**
- **Select Teams:** Choose the batting and bowling teams from the dropdown menus.
- **Select City:** Choose the match city from the available options.
- **Enter Current Score:** Input the current runs scored by the batting team.
- **Overs Completed:** Input the number of overs completed (must be greater than 5).
- **Wickets Fallen:** Input the number of wickets fallen.
- **Bowler Wickets:** Input the number of wickets taken by the current bowler.
- **Runs in Last 5 Overs:** Input the runs scored in the last 5 overs.
2. **Predict Score:** Click the "Predict Score" button to receive the predicted final score based on the input parameters.

## Model Evaluation

After training, the model's performance is evaluated using:
- **R² Score:** Indicates the proportion of variance in the dependent variable predictable from the independent variables.
- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions, without considering their direction.

These metrics are printed in the console upon running the training script.







