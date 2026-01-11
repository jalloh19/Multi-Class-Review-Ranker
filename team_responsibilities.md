# Team Responsibilities: Multi-Class Review Ranker

This document outlines the specific tasks for each team member to ensure efficient collaboration

## Shared Responsibilities
**Owner:** All Team Members
**File:** `notebooks/01_data_preparation.ipynb`
*   **Data Loading:** Ensure `Amazon_Reviews.csv` loads correctly.
*   **Preprocessing Logic:** Define the `clean_text` function in `src/preprocessing.py`. This is the single source of truth for text cleaning.
*   **Data Splitting:** Run the data preparation notebook to generate `data/processed/train.csv` and `data/processed/test.csv`.
    *   *Constraint:* Everyone MUST use these exact CSV files for training to ensure model comparison is valid.

---

## 🧑‍💻 Student 1: Jalloh (Naive Bayes Model)
**Workspace:** `notebooks/02_model_bayes.ipynb`
**Goal:** Develop a baseline probabilistic model using Naive Bayes.

### Tasks:
1.  **Load Data:** Read the shared `train.csv` and `test.csv`.
2.  **Feature Extraction:** Implement `CountVectorizer` or `TfidfVectorizer`.
3.  **Model Training:**
    *   Train a `MultinomialNB` model.
    *   Perform Hyperparameter tuning (e.g., `alpha` smoothing).
4.  **Evaluation:**
    *   Generate a Classification Report (Precision, Recall, F1 for each of the 5 classes).
    *   Create a Confusion Matrix plot.
5.  **Deliverable:**
    *   Save the final pipeline (Vectorizer + Model) to `saved_models/bayes_model.pkl`.
    *   Update `src/predict.py` (if applicable) or provide the loading logic.

---

## 🧑‍💻 Student 2: Madiou (SVM Model)
**Workspace:** `notebooks/03_model_svm.ipynb`
**Goal:** Develop a robust discriminator model using Support Vector Machines.

### Tasks:
1.  **Load Data:** Read the shared `train.csv` and `test.csv`.
2.  **Feature Extraction:** Implement `TfidfVectorizer` (match Jalloh's generic approach or try advanced N-grams).
3.  **Model Training:**
    *   Train a `LinearSVC` or `SVC(kernel='linear')`.
    *   *Important:* If using `SVC`, set `probability=True` to enable confidence scores for the frontend.
    *   Tune regularization parameter `C`.
4.  **Evaluation:**
    *   Generate Classification Report.
    *   Compare accuracy against the Bayes baseline.
5.  **Deliverable:**
    *   Save the final pipeline to `saved_models/svm_model.pkl`.

---

## 🎨 Mustafa: Frontend Developer (Streamlit App)
**Workspace:** `app/app.py` & `src/utils.py`
**Goal:** Create a user-friendly Dashboard for inference and comparison.

### 1. Detailed UI/UX Requirements
#### Core Features
*   **Model Selection & Input**:
    *   **Input Area**: A large text box for users to paste or type review text.
    *   **Model Selector**: A way to choose the analysis mode: `Naive Bayes` only, `SVM` only, or `Both` (Default - Comparison Mode).
    *   **Action**: An "Analyze" button to trigger the inference.

#### Analysis Results & Comparison
*   **Single Model Mode**:
    *   **Predicted Class**: Map star ratings to textual sentiment (e.g., 5 Stars = "Very Good / Delighted 🤩", 1 Star = "Very Bad / Angry 😠").
    *   **Confidence Information**: Show a probability bar for the specific emotion/class.
    *   **Sentiment Interpretation**: e.g., "This review expresses significant dissatisfaction."
*   **Comparison Mode (Both)**:
    *   **Side-by-Side Cards**: One column for Bayes, one for SVM.
    *   **Agreement Check**: Highlight if models agree or disagree.
    *   **Visualization**: A bar chart comparing confidence scores.

#### Pipeline Visualization ("Under the Hood")
*   An expandable section (`st.expander`) revealing:
    *   **Raw Input**: Exact text entered.
    *   **Preprocessing Steps**: Text after cleaning (must import `clean_text` from `src.preprocessing`).

### 2. Deployment Tasks
*   **Prepare Environment**: Update `requirements.txt` to include `streamlit`, `scikit-learn`, `joblib`, etc.
*   **Local Testing**: Ensure `streamlit run app/app.py` works without errors.
*   **Cloud Deployment** (Streamlit Cloud):
    *   Push final code to GitHub.
    *   Connect repository to Streamlit Cloud.
    *   Verify the live application link.

### 3. Deliverables
*   A fully functional, responsive `app/app.py` handling missing models gracefully.
*   A live deployment URL (optional but recommended).

---

## 🗓️ Workflow & Timeline

### Phase 1: Setup
*   **All:** Agree on `clean_text` logic in `src/preprocessing.py`.
*   **All:** Run `01_data_preparation.ipynb` to create the processed data.

### Phase 2: Independent Development (Days 2-4)
*   **Jalloh:** Works in `notebooks/02_model_bayes.ipynb`.
*   **Madiou:** Works in `notebooks/03_model_svm.ipynb`.
*   **Frontend:** Works in `app/app.py` using **Mock Data** (fake predictions) to build the UI layout.

### Phase 3: Integration
*   **Models:** Jalloh & Madiou save their best models to `saved_models/`.
*   **Frontend:** Frontend dev updates `app.py` to load the real `.pkl` files and removes mock data.
*   **All:** Test the full pipeline: Input -> Clean -> Predict -> Display.
