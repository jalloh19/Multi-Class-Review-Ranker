# Data Science Project Structure

## Project Organization

- `app/`           : Streamlit web application
- `data/`          : Data storage
    - `raw/`       : Immutable original data
    - `processed/` : Cleaned and preprocessed data
- `docs/`          : Documentation, references, and reports
- `notebooks/`     : Jupyter notebooks for exploration and prototyping
- `saved_models/`  : Trained model artifacts
- `src/`           : Source code for use in this project
    - `preprocessing.py` : Scripts to clean and transform data
    - `train.py`         : Scripts to train models
    - `evaluate.py`      : Scripts/functions to evaluate model performance
    - `utils.py`         : Helper functions
- `tests/`         : Unit tests
- `requirements.txt`: Python dependencies
- `README.md`      : Project overview

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app/app.py
   # or if using streamlit
   # streamlit run app/app.py
   ```
