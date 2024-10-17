# Srentimental_Analysis
Sentiment Analysis on IMDb Movie Reviews

1. Problem Statement
The goal of this project is to develop and evaluate different machine learning and deep learning models to classify movie reviews from the IMDb dataset as either positive or negative. 
By comparing the performance of models like RNN, LSTM, RoBERTa, DistilBERT, and a hybrid model, this project aims to find the best approach to achieve high accuracy and robust sentiment analysis.

2.Brief Description of Approach
This project employs a variety of machine learning and deep learning models for sentiment analysis, including:

Recurrent Neural Networks (RNN)
Long Short-Term Memory Networks (LSTM)
Pretrained Transformers like RoBERTa and DistilBERT
A hybrid model combining CNN, RoBERTa, DistilRoBERTa, and BiLSTM with Attention mechanisms for enhanced performance.
Steps Taken:
Data Preprocessing: The IMDb movie reviews dataset is tokenized, padded, and processed to ensure compatibility with various models.
Word embeddings like Word2Vec and GloVe are utilized to improve textual representation.

Model Training: Several models, including RNN, LSTM, and transformer-based models (e.g., RoBERTa, DistilBERT), are trained and evaluated. 
Hyperparameter tuning and optimization are performed to achieve the best results.

Evaluation: Models are evaluated using metrics like accuracy, loss, precision, recall, and F1-score, 
with plots provided to visualize the performance across various epochs.

User Input: The notebook allows users to input their own data (CSV or manual text input) to perform sentiment analysis using the trained models

3.How to run

Here are the instructions on how to run the code as plain text:

---

**How to Run the Code:**

1. **Clone the repository:**
   - Run the following command to clone the project repository:
     ```
     git clone https://github.com/your-repo/sentiment-analysis.git
     ```

2. **Navigate to the project directory:**
   - Move into the project folder using:
     ```
     cd sentiment-analysis
     ```

3. **Install dependencies:**
   - Make sure you have Python version 3.7 or later installed.
   - Install all the required dependencies by running:
     ```
     pip install -r requirements.txt
     ```

4. **Run the Jupyter Notebook:**
   - Launch Jupyter Notebook by running:
     ```
     jupyter notebook Sentimental_analysis_IMDBReviews_Thackshana.ipynb
     ```

5. **Execute the notebook cells:**
   - Open the notebook in the Jupyter interface.
   - Run the cells in sequence, following the steps to preprocess the data, train the models, and evaluate performance.

6. **Train and Evaluate Models:**
   - The notebook will guide you through the training of models like RNN, LSTM, RoBERTa, and the custom hybrid model.
   - You can choose different models for evaluation and compare their performance.

7. **Perform Sentiment Analysis:**
   - Use the provided functions in the notebook to analyze sentiment on new movie review data or upload a CSV file containing reviews.

---

These are the steps in text form that will allow anyone to run the notebook and reproduce the results.
