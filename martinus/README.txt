## Linear, Lasso, and Ridge Regression

## Requirements
python3==3.4.3
pandas==0.13.1
scikit-learn==0.17.1
numpy==1.11.1
scipy==0.18.1

## Training process
1. Please make sure that training data `data_split.pkl` is available inside `<project_directory>`
2. Please make sure users data `users_cleaned_binary.csv` and words data `words_cleaned_binary.csv` are available inside `<project_directory>/data`
3. Change directory to `<project_directory>/martinus`
4. Train the individual models
   - Linear Regression (fully combined): `python3 linear_regression.py`
   - Ridge (fully combined): `python3 ridge.py`
   - Lasso (fully combined): `python3 lasso.py`

5. The MSE will be displayed in the console after the training process finished
6. After each training, two pickle files will be generated. One is the trained models, the other is the predicted models which will be used for the neural network ensemble training.