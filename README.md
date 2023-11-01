# Master-Thesis-Data-Drift

This repository contains a collection of functions for text data preprocessing, TF-IDF vectorization, cosine similarity calculation, data visualization, and analysis, aimed at text similarity computations and comparison.

Table of Contents
General Functions
Text Preprocessing
TF-IDF Calculation
Cosine Similarity Calculation
Data Visualization
Word2Vec Analysis
Thresholds Analysis
General Functions <a name="general-functions"></a>
The repository provides several general functions for handling text data:

create_dataframe(path): Creates a Pandas DataFrame from text files located at a specified path.
preprocess_text(df, column): Preprocesses text data in a DataFrame for a specific column.
tfidf(train, column_train, test, column_test): Calculates TF-IDF vectorization for training and test data.
calculate_cosine_similarity(train, test): Computes the cosine similarity between two sets of vectors.
heatmapvis(argument): Generates a heatmap plot for better visualization.
histogramvis(argument): Produces a histogram plot for data distribution visualization.
top10(cosinesimilarity): Determines the top 10 most similar documents based on cosine similarity.
percentile(cosinesimilarity): Calculates the 10th, 50th, and 75th percentiles for the dataset's cosine similarity distribution.
word2vec(df, column): Computes Word2Vec representation for a given dataset.
Text Preprocessing <a name="text-preprocessing"></a>
The preprocess_text() function takes a DataFrame and preprocesses a specific column. It performs tasks such as removing special characters, numbers, converting text to lowercase, removing single alphabets, tokenization, removing stopwords, lemmatization, and more.

TF-IDF Calculation <a name="tf-idf-calculation"></a>
The tfidf() function calculates TF-IDF vectorization for the provided training and test datasets using the TfidfVectorizer from scikit-learn.

Cosine Similarity Calculation <a name="cosine-similarity-calculation"></a>
The calculate_cosine_similarity() function computes the cosine similarity between two sets of vectors, useful for text similarity analysis.

Data Visualization <a name="data-visualization"></a>
The functions heatmapvis() and histogramvis() help visualize cosine similarity through heatmap plots and histogram distributions for better data interpretation.

Word2Vec Analysis <a name="word2vec-analysis"></a>
The repository also includes functions for Word2Vec analysis. word2vec() computes Word2Vec representations for a given dataset.

Thresholds Analysis <a name="thresholds-analysis"></a>
Finally, the analysis involves loading previously saved thresholds and creating box plots for mean cosine similarities obtained from both TF-IDF and Word2Vec analysis for different test datasets.

For further details on usage and functions, please refer to the provided code.




