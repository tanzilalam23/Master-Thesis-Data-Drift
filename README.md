# Project Overview

This repository comprises a suite of functions dedicated to text data preprocessing, TF-IDF vectorization, cosine similarity computation, data visualization, and analysis. The primary focus is on text similarity calculations and comparisons.

## Table of Contents
1. [General Functions](#general-functions)
2. [Text Preprocessing](#text-preprocessing)
3. [TF-IDF Calculation](#tf-idf-calculation)
4. [Cosine Similarity Calculation](#cosine-similarity-calculation)
5. [Data Visualization](#data-visualization)
6. [Word2Vec Analysis](#word2vec-analysis)
7. [Thresholds Analysis](#thresholds-analysis)

---

## General Functions <a name="general-functions"></a>

The repository offers a set of general functions specifically designed for managing text data:

- `create_dataframe(path)`: Creates a Pandas DataFrame from text files located at a specified path.
- `preprocess_text(df, column)`: Handles text data preprocessing within a DataFrame for a specified column.
- `tfidf(train, column_train, test, column_test)`: Computes TF-IDF vectorization for both training and test data.
- `calculate_cosine_similarity(train, test)`: Determines the cosine similarity between two sets of vectors.
- `heatmapvis(argument)`: Generates a heatmap plot for improved data visualization.
- `histogramvis(argument)`: Produces a histogram plot for visualizing data distributions.
- `top10(cosinesimilarity)`: Identifies the top 10 most similar documents based on cosine similarity.
- `percentile(cosinesimilarity)`: Calculates the 10th, 50th, and 75th percentiles for the distribution of cosine similarity within the dataset.
- `word2vec(df, column)`: Computes Word2Vec representation for a given dataset.

---

## Text Preprocessing <a name="text-preprocessing"></a>

The `preprocess_text()` function operates on a DataFrame and performs various preprocessing tasks, including special character and number removal, text conversion to lowercase, elimination of single alphabets, tokenization, stopword removal, lemmatization, and more.

---

## TF-IDF Calculation <a name="tf-idf-calculation"></a>

The `tfidf()` function calculates TF-IDF vectorization for the provided training and test datasets using the TfidfVectorizer from scikit-learn.

---

## Cosine Similarity Calculation <a name="cosine-similarity-calculation"></a>

The `calculate_cosine_similarity()` function computes the cosine similarity between two sets of vectors, a crucial tool for text similarity analysis.

---

## Data Visualization <a name="data-visualization"></a>

The repository includes functions like `heatmapvis()` and `histogramvis()` for visualizing cosine similarity through heatmap plots and histogram distributions, enhancing data interpretation.

---

## Word2Vec Analysis <a name="word2vec-analysis"></a>

The repository also offers functions dedicated to Word2Vec analysis. `word2vec()` computes Word2Vec representations for a given dataset.

---

## Thresholds Analysis <a name="thresholds-analysis"></a>

The analysis involves loading previously saved thresholds and creating box plots for mean cosine similarities obtained from both TF-IDF and Word2Vec analysis for different test datasets.

For comprehensive details on usage and functions, please refer to the provided code.
