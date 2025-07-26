# Hands-On NLP Project

A practical, end-to-end Natural Language Processing (NLP) pipeline for sentiment analysis using the IMDB movie reviews dataset. This project demonstrates data cleaning, preprocessing, feature extraction, model training, and prediction using Python's most popular NLP and machine learning libraries.

---

## 🚀 Features
- **Data Loading & Exploration**: Loads and inspects the IMDB dataset (50,000 labeled reviews)
- **Text Preprocessing**: Cleans text, removes stopwords, tokenizes, and lemmatizes using NLTK and spaCy
- **Feature Engineering**: Converts text to numerical features with TF-IDF
- **Model Training**: Trains a Logistic Regression classifier for sentiment prediction
- **Evaluation**: Shows class balance and model performance
- **Custom Prediction**: Predicts sentiment for new, unseen reviews

---

## 🗂️ Project Structure
- `main.ipynb` — Main Jupyter notebook with the full workflow
---

## 🛠️ Setup Instructions
1. **Clone the repository and navigate to the project directory**
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install pandas spacy scikit-learn matplotlib seaborn nltk
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```
4. **Ensure `IMDB Dataset.csv` is in the project root**

---

## 📒 Usage
1. **Open the notebook**
   ```bash
   jupyter notebook main.ipynb
   ```
2. **Run all cells** to:
   - Download NLTK and spaCy resources
   - Load and explore the dataset
   - Preprocess and clean the text
   - Train/test a sentiment classifier
   - Predict sentiment for custom reviews

---

## 🧹 Example Workflow
```python
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Load data
df = pd.read_csv('IMDB Dataset.csv')

# Preprocess text (clean, tokenize, remove stopwords, lemmatize)
# ... see notebook for full code ...

# Train/test split, TF-IDF, model training
# ... see notebook for full code ...

# Predict sentiment for new review
review = "This movie was amazing!"
prediction = predict_sentiment(review)
print(prediction)  # Output: Positive or Negative
```

---

## 📝 Example Output
```
Review: 'This movie was absolutely fantastic! The acting was superb and the plot was gripping.'
Predicted Sentiment: Negative

Review: 'I was so bored throughout the entire film. It was a complete waste of time and money.'
Predicted Sentiment: Negative

Review: 'The film was okay, not great but not terrible either. Some parts were good.'
Predicted Sentiment: Positive
```

---

## 📚 Dependencies
- pandas
- numpy
- nltk
- spacy
- scikit-learn
- matplotlib
- seaborn

---

## 💡 Notes
- The notebook samples 500 reviews for faster processing. For full dataset training, remove or adjust the sampling line.
- Make sure to download all required NLTK and spaCy resources before running the full pipeline.

---

## 🤝 Contributing
Pull requests and suggestions are welcome!
