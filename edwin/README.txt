## Random Forest Full, Random Forest by Artist, and Linear Regression by Artist


## Requirements
- Python 3
- Scikit-learn
- Pandas
- Numpy


## Steps
1. Open `<path_to_project_dir>/edwin/util/data_util.py`
2. Change the data set that you want to use for training
    - The name of the data set depends on the settings that you use when running `<path_to_project_dir>/data_split_util.py`
    - By default, there should be set A, B, AB (combined data from A and B), and C
    - This can be changed under these 2 variables (both values should refer to the dictionary key used in `<path_to_project_dir>/data_split_util.py`:
        - X_SET_TO_BE_PREPROCESSED
        - Y_SET_TO_BE_PREPROCESSED

3. To train the model based on the preprocessed training data:
    - Linear Regression by Artist: `python3 <path_to_project_dir>/edwin/lr_by_artist/train_lr_by_artist.py`
    - Random Forest by Artist: `python3 <path_to_project_dir>/edwin/rf_by_artist/train_rf_by_artist.py`
    - Random Forest Full Data Features: `python3 <path_to_project_dir>/edwin/rf_full/train_rf_full.py`

4. To predict the rating a dataset (A, B, AB, or C):
    - Change the dataset which you want to predict the ratings for
        - In the "if __name__ == '__main__'" part, there should be this line of code:
            - predictions = predict_ratings_and_calculate_mse(<X_SET>, <Y_SET>)
            - Change the value of <X_SET> and <Y_SET> accordingly
            - Available value: X_A, Y_A, X_B, Y_B, X_C, Y_C

    - Linear Regression by Artist: `python3 <path_to_project_dir>/edwin/lr_by_artist/predict_lr_by_artist.py`
    - Random Forest by Artist: `python3 <path_to_project_dir>/edwin/rf_by_artist/predict_rf_by_artist.py`
    - Random Forest Full Data Features: `python3 <path_to_project_dir>/edwin/rf_full/predict_rf_full.py`

5. The MSE and/or RMSE results should be displayed in the console
6. There should be some zipped pickle files that contains the predicted ratings which will be used by the ensemble models