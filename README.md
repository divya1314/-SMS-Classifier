# SMS Spam Classification Project

## Project Overview

This project aims to develop a text classification model that can classify SMS messages as either spam or non-spam (ham) using data science techniques in Python. The project was undertaken as part of an internship at Bharat Intern.

## Dataset

The dataset used for this project is the "SMS Spam Collection Dataset," which is publicly available on Kaggle. It can be accessed through the following link: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). This dataset consists of a collection of 5,574 SMS messages tagged as spam or ham.

## Methodology

1. **Data Loading and Exploration**:
   - The dataset was loaded into a Pandas DataFrame and the first few rows were inspected to understand its structure.

2. **Data Preprocessing**:
   - The labels were converted into a binary format: 'ham' as 0 and 'spam' as 1.
   - Text data was cleaned by removing non-alphabetic characters, converting to lowercase, and removing stopwords.
   - Words were stemmed to their root forms using the Porter Stemmer.

3. **Feature Extraction**:
   - Text data was converted into numerical features using the TfidfVectorizer. This technique transforms the text into a matrix of TF-IDF features, capturing the importance of words relative to the document and the corpus.

4. **Model Training**:
   - The dataset was split into training and test sets with an 80-20 ratio.
   - A Multinomial Naive Bayes model, suitable for text classification, was trained on the training data.

5. **Model Evaluation**:
   - The model's performance was evaluated on the test set using accuracy, confusion matrix, and classification report metrics.
   - The accuracy score, precision, recall, and F1-score were calculated to assess the model’s effectiveness.
   - A confusion matrix was plotted to visualize the true positives, false positives, true negatives, and false negatives.

## Results

- **Accuracy**: The model achieved a high accuracy score, indicating its effectiveness in classifying SMS messages as spam or ham.
- **Confusion Matrix**: The confusion matrix showed a low number of false positives and false negatives, confirming the model’s robustness.
- **Classification Report**: Precision, recall, and F1-scores were high, further demonstrating the model’s reliability.

## Conclusion

The SMS spam classifier developed in this project performs well in distinguishing between spam and non-spam messages. The use of text preprocessing, feature extraction with TF-IDF, and the Multinomial Naive Bayes algorithm proved to be an effective approach for this classification task. This project not only enhances the understanding of natural language processing and machine learning techniques but also provides a practical tool for spam detection.

## Future Work

To further improve the model, future work could explore:
- Using more advanced preprocessing techniques.
- Experimenting with different feature extraction methods.
- Trying more complex models such as Support Vector Machines, Random Forests, or deep learning models like LSTM or BERT.
- Performing hyperparameter tuning to optimize the model's performance.

This project was an excellent learning experience and contributed significantly to practical skills in data science and machine learning.
