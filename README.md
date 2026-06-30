# amazon_sentiment_prediction
This project is about predicting the sentiment of the reviews given by the customer.
It is a multi classification model that required to predict 3 different classes.
1. Positive
2. Negative
3. Neutral

We have trained ML models (random forest, SVM, Logistic regression) on the dataset.
Built API to test the model predictions.

To run the main.py from api/ and test predictions. run the below command
uvicorn api.main:app --port 8080

