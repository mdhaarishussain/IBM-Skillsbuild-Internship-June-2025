💼 Employee Salary Classification | Capstone Project
====================================================

<a href="https://ibb.co/KcR1Z8Wh"><img src="https://i.ibb.co/NgQ4hGKC/Screenshot-2025-07-23-031743.png" alt="Screenshot-2025-07-23-031743" border="0"></a>

An Advanced ML model to predict whether an employee earns >$50K or ≤$50K. A Project by Md Haaris Hussain 🎓 EDUNET FOUNDATION - IBM SKILLSBUILD AI INTERNSHIP (JUNE 2025)

✨ Project Overview
------------------

This project aims to develop a robust machine learning model capable of predicting whether an individual's annual income exceeds $50,000 based on various demographic and employment-related features. The entire workflow, from data ingestion and rigorous preprocessing to model training, evaluation, and interactive deployment, is meticulously covered. The goal is to provide a user-friendly Streamlit web application for both single and batch predictions, making the power of predictive analytics accessible.

### 🎯 Key Goals:

-   **Data Exploration & Cleaning:** Load and thoroughly explore the Adult Census Income dataset, identifying and handling missing values, outliers, and inconsistencies.

-   **Feature Engineering:** Transform raw categorical features into a numerical format suitable for machine learning algorithms (e.g., using Label Encoding).

-   **Model Development:** Train and fine-tune multiple classification algorithms to find the optimal model for income prediction.

-   **Model Evaluation:** Assess the performance of trained models using appropriate metrics (e.g., accuracy, precision, recall, F1-score).

-   **Interactive Deployment:** Deploy the best-performing model as an intuitive Streamlit web application, enabling real-time and batch predictions.

📁 Project Structure
--------------------

```
.
├── adult 3 (1).csv             # The raw Adult Census Income dataset
├── app.py                      # Streamlit web application for predictions
├── best_model.pkl              # Trained machine learning model (GradientBoostingClassifier)
├── employee_salary_prediction.ipynb # Jupyter Notebook for EDA, preprocessing, and model training
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── Readme.md                   # This README file
└── requirements.txt            # Python dependencies for the project

```

💾 Dataset
----------

The project leverages the **Adult Census Income Dataset**, a widely recognized benchmark dataset for classification tasks, sourced from the UCI Machine Learning Repository.

-   **File:**  `adult 3 (1).csv`

-   **Description:** This dataset contains information extracted from the 1994 Census database, encompassing various demographic and employment-related features for individuals. The primary target variable is `income`, indicating whether an individual's annual income is `<=50K` or `>50K`.

### Key Features:

-   `age`: Continuous.

-   `workclass`: Categorical (e.g., Private, Self-emp-not-inc, Local-gov, ?).

-   `fnlwgt`: Continuous (final weight - the number of people the census believes the entry represents).

-   `education`: Categorical (e.g., HS-grad, Some-college, Bachelors).

-   `educational-num`: Continuous (numerical representation of education level).

-   `marital-status`: Categorical.

-   `occupation`: Categorical (e.g., Tech-support, Craft-repair, Sales, ?).

-   `relationship`: Categorical.

-   `race`: Categorical.

-   `gender`: Binary (Male, Female).

-   `capital-gain`: Continuous.

-   `capital-loss`: Continuous.

-   `hours-per-week`: Continuous.

-   `native-country`: Categorical (e.g., United-States, Mexico, ?).

-   `income`: **Target Variable** (Categorical: `<=50K`, `>50K`).

🧠 Model
--------

The core of this project is a machine learning model trained to classify income levels.

-   **Model File:**  `best_model.pkl`

-   **Algorithm:** Based on the `employee_salary_prediction.ipynb` notebook, the best-performing model chosen is a `GradientBoostingClassifier`. This model is serialized using `joblib` for efficient loading and prediction within the web application.

-   **Preprocessors:** The `app.py` also includes logic to load or create default `LabelEncoder` instances for categorical features, ensuring consistent preprocessing between training and inference.

🚀 Application
--------------

The project includes an interactive web application built with Streamlit, allowing users to make predictions effortlessly.

-   **Application File:**  `app.py`

-   **Live Application:**  <https://employeesalarypredictionedunet.streamlit.app/>

-   **Features:**

    -   **Single Prediction:** Input individual features through a user-friendly form to get an instant salary class prediction.

    -   **Batch Prediction:** Upload a CSV file containing multiple entries to receive predictions and confidence scores for each, with an option to download the results.

    -   **Modern UI:** Features a dark mode theme with a clean and responsive design.

🛠️ Setup and Installation
--------------------------

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```
    git clone <repository_url>
    cd employee-salary-classification

    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    ```

3.  **Install the required dependencies:**

    ```
    pip install -r requirements.txt

    ```

    The `requirements.txt` file contains:

    ```
    pandas
    scikit-learn
    matplotlib
    seaborn
    streamlit
    joblib

    ```

4.  **Ensure model and data files are present:** Make sure `adult 3 (1).csv` and `best_model.pkl` are in the root directory of the project.

🏃 Usage
--------

To launch the Streamlit web application:

```
streamlit run app.py

```

This command will open the application in your default web browser. You can then interact with the UI to perform single or batch predictions.

🤝 Contributing
---------------

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.

2.  Create a new branch (`git checkout -b feature/your-feature-name`).

3.  Make your changes.

4.  Commit your changes (`git commit -m 'Add new feature'`).

5.  Push to the branch (`git push origin feature/your-feature-name`).

6.  Open a Pull Request.

📄 License
----------

This project is open-sourced under the MIT License. See the `LICENSE` file for more details.

<p align="center">Developed by <strong>Md Haaris Hussain</strong></p> <p align="center">Powered by IBM SkillsBuild & Edunet Foundation | Built with Streamlit</p>