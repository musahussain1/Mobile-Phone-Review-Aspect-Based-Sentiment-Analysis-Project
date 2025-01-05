# Mobile Phone Review Aspect-Based Sentiment Analysis Project

**Repository URL**: [GitHub Repository](https://github.com/musahussain1/Mobile-Phone-Review-Aspect-Based-Sentiment-Analysis-Project)

---

## Overview

This repository contains the code and data for the Mobile Phone Review Aspect-Based Sentiment Analysis Project. The project focuses on analyzing customer reviews of mobile phones to extract specific aspects (e.g., battery, screen, case) and predict sentiments associated with those aspects.

Aspect-Based Sentiment Analysis (ABSA) provides a more detailed evaluation of customer feedback by categorizing sentiments by topics, overcoming the limitations of traditional sentiment analysis.

---

## Project Objectives

1. **Improving Sentiment Analysis**: Enhance the performance of sentiment analysis systems using Aspect-Based Sentiment Analysis (ABSA) techniques.
2. **Topic Extraction Using Pretrained LLMs**: Leverage large language models (e.g., Facebook's BART) to identify topics in customer reviews.
3. **Integrated Sentiment and Topic Prediction**: Develop machine learning models to predict both sentiment and topics for each review.

For detailed objectives, refer to the [Project Objectives Document](./Project Objectives.docx).

---

## Dataset

The dataset used for this project contains Amazon product reviews of mobile phones. Key columns include:

- **Rating**: Product rating (1 to 5).
- **Title**: Title of the user review.
- **Text**: Body of the review.
- **Timestamp**: Date and time of the review.
- **Verified Purchase**: Whether the purchase was verified.
- **Helpful Votes**: Number of helpful votes the review received.

For additional details, check the [Data Information Document](./Data Information.docx).

---

## Key Features

1. **Preprocessing**: Data cleaning, tokenization, lemmatization, and removal of stopwords.
2. **Topic Modeling**: Utilized LDA and large language models like Facebook BART for topic extraction.
3. **Sentiment Analysis**: Combined `TextBlob` and machine learning algorithms to classify sentiments as positive, negative, or neutral.
4. **Integrated Prediction**: Machine learning models (e.g., Random Forest, Logistic Regression) trained to predict both sentiments and topics simultaneously.

---

## Repository Structure

- **Code/**
  - `Aspect_Based_Sentiment_Analysis.py`: Python scripts for data preprocessing, topic modeling, sentiment analysis, and model training.
  - `desktop.ini`: System-generated file.
- **Data/**
  - `Cell_Phones_and_Accessories.jsonl`: Raw data of the all mobile phone reviews
  - `loaded_data.csv`: Cleaned and preprocessed dataset.
  - `topic_data.csv`: Dataset with predicted topics.
- `LICENSE`: Licensing information for the project.
- `README.md`: This file, documenting the repository structure and usage.

---

## Setup and Usage

### Prerequisites
Ensure the following Python libraries are installed:
- `nltk`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `textblob`, `scikit-learn`, `transformers`, `wordcloud`, `tqdm`, `imblearn`

Install them using:
```bash
pip install nltk pandas numpy matplotlib seaborn textblob scikit-learn transformers wordcloud tqdm imblearn
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/musahussain1/Mobile-Phone-Review-Aspect-Based-Sentiment-Analysis-Project.git
   ```
2. Navigate to the directory:
   ```bash
   cd Mobile-Phone-Review-Aspect-Based-Sentiment-Analysis-Project
   ```
3. Execute the Python script for analysis:
   ```bash
   jupyter notebook Code/Aspect_Based_Sentiment_Analysis.ipynb
   ```
