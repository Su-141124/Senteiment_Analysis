Restaurant Review Sentiment Analysis:-
This project implements a sentiment analysis model that classifies restaurant reviews as Positive, Negative, or Neutral using machine learning and natural language processing (NLP) techniques.
Overview:-
The notebook reads a dataset of restaurant reviews, cleans and vectorizes the text using TF-IDF, and applies logistic regression to predict sentiment. It demonstrates a full machine learning pipeline including training, testing, and prediction.

Technologies Used:-
- Python
- Jupyter Notebook
- pandas
- scikit-learn
- TfidfVectorizer
- LogisticRegression

Workflow:-
1. Load and explore review dataset
2. Preprocess text (cleaning, removing stopwords, etc.)
3. Split the data into training and test sets
4. Transform text data using TF-IDF vectorization
5. Train a Logistic Regression model on the training data
6. Evaluate the model using accuracy, precision, recall, and F1-score
7. Make predictions on new reviews

Example:-
Input: "The food was delicious and the service was excellent!"
Output: Positive

Future Improvements:-
- Add more data to improve accuracy
- Implement deep learning models like LSTM or BERT
- Deploy as a web app using Streamlit or Flask
- Include advanced text preprocessing (stemming, lemmatization)
