**Text Classification for Cyberbullying Detection**


### **1. Problem Definition**
The objective is to create a robust system capable of distinguishing between neutral text and various forms of cyberbullying. The model is trained to recognize specific categories of abuse, such as those based on religion, age, ethnicity, and gender.

### **2. Exploratory Data Analysis (EDA)**
Before modeling, the data undergoes a thorough analysis to understand its structure:
* **Class Distribution:** Checking for imbalances between bullying and non-bullying categories.
* **Text Statistics:** Analyzing word counts and sentence lengths.
* **Word Clouds:** Identifying high-frequency terms associated with different types of cyberbullying.

### **3. Data Preprocessing Pipeline**
The text is cleaned to ensure the models focus on meaningful patterns:
* **Normalization:** Converting text to lowercase and expanding contractions.
* **Noise Removal:** Stripping out special characters, URLs, and punctuation.
* **Tokenization & Lemmatization:** Breaking sentences into individual words and reducing them to their base dictionary forms.
* **Advanced Cleaning:** Using the SymSpell algorithm for spell correction and TextBlob for initial sentiment analysis.

### **4. Feature Engineering**
To make the text readable for machine learning algorithms, it is converted into numerical vectors using:
* **TF-IDF Vectorization:** Assigning weights to words based on their importance across the dataset.
* **Chi-Square (Chi2) Selection:** Selecting the most statistically significant features to improve model efficiency and reduce noise.

### **5. Model Implementation & Persistence**
A variety of machine learning algorithms are implemented, compared, and tuned. To ensure the trained models can be reused without retraining, the best-performing versions are saved as **Pickle (.pkl)** files:

* **Decision Tree:** `best_dt_model.pkl`
* **K-Nearest Neighbors:** `best_knn_model_tuned.pkl`
* **Logistic Regression:** `best_lr_model.pkl`
* **Naïve Bayes:** `best_nb_model.pkl`
* **Random Forest:** `best_rf_model.pkl`
* **Support Vector Machine:** `best_svm_model.pkl`

### **6. Evaluation & Deployment**
The performance is measured using **Accuracy and F1-Score** to ensure accurate identification of bullying instances. These saved models are integrated into a **Gradio-based Graphical User Interface (GUI)**, allowing users to input text and receive instant classification results.

full code in : Cyberbullying_Detection.ipynb
