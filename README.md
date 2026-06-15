# UCLA Cross-Regional Enrollment Analytics via Unsupervised Learning

An unsupervised machine learning pipeline designed to segment and evaluate high school enrollment pipelines for the University of California, Los Angeles (UCLA). By utilizing $K$-Means clustering and Singular Value Decomposition (SVD), this project uncovers localized behavioral conversion trends and demographic disparities across 1,400+ in-state and out-of-state feeder institutions.

---

## 📂 Repository Structure

* **`cleaned_data/`**
    * `parseable_cali_gender_data.csv`: Restructured, flat matrix of California high school cohorts ready for numerical operations.
    * `parseable_NONCALI_gender_data.csv`: Restructured, flat matrix of out-of-state institutional pipelines.
* **`data/`**
    * `GENDER_CALI_HS.csv`: Raw, tab-delimited raw enrollment export from the UC Information Center (In-State).
    * `GENDER_NONCALI_HS.csv`: Raw, tab-delimited raw enrollment export from the UC Information Center (Out-of-State).
* **`notebooks/`** ⭐⭐
    * `notebook.ipynb`: This is where you find **documentation and explanations.** Core execution environment containing hyperparameter tuning (Elbow Method), algorithmic iterations, SVD projections, and strategic data visualizations.
* **`src/`**
    * `data_pipeline.py`: Custom preprocessing module managing data sanitization, schema flattening (`create_parseable_df`), and demographic funnel metric engineering (`compute_funnels`).
    * `models.py`: First-principles implementations of core mathematical modules, including `StandardScaler` and `KMeans`.

---

