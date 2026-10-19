import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from typing import List, Dict, Tuple, Union, Optional, Callable

class BM25:
    """
    BM25 implementation for document similarity search
    """
    def __init__(self, 
                corpus: List[str], 
                tokenizer: Optional[Callable] = None,
                k1: float = 1.5,
                b: float = 0.75,
                epsilon: float = 0.25,
                normalize_scores: bool = True):
        """
        Initialize BM25 with corpus of documents
        
        Args:
            corpus: List of document strings
            tokenizer: Custom tokenizer function (optional)
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            epsilon: Small value to prevent division by zero and negative IDF
            normalize_scores: Whether to normalize scores to 0-100 range
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty")
            
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.normalize_scores = normalize_scores
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(tokenizer=tokenizer, min_df=1)
        self.doc_term_matrix = self.vectorizer.fit_transform(corpus)
        self.vocab_size = len(self.vectorizer.get_feature_names_out())
        
        # Precompute document lengths and average document length
        self.doc_len = self.doc_term_matrix.sum(1).A1
        self.avdl = np.mean(self.doc_len) if len(self.doc_len) > 0 else 0
        
        # Precompute IDF values
        self.df = np.squeeze(np.asarray(self.doc_term_matrix.sum(0)))
        # Modified IDF calculation to avoid negative IDF values
        self.idf = np.log(1 + (len(self.corpus) - self.df + 0.5) / (self.df + 0.5 + self.epsilon))
    
    def _normalize_to_0_100(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to 0-100 range
        
        Args:
            scores: Array of raw BM25 scores
            
        Returns:
            Array of scores normalized to 0-100 range
        """
        if np.all(scores == 0):
            return np.zeros_like(scores)
            
        # Min-Max normalization
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # If all scores are the same, return 100 for matches, 0 for no matches
        if max_score == min_score:
            if max_score > 0:
                return np.ones_like(scores) * 100
            else:
                return np.zeros_like(scores)
                
        # Scale scores to 0-100
        normalized = (scores - min_score) * 100 / (max_score - min_score)
        return normalized
        
    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculate BM25 scores for a query against all documents in corpus
        
        Args:
            query: The search query string
            
        Returns:
            Array of scores for each document (0-100 if normalize_scores is True)
        """
        if not query or not query.strip():
            return np.zeros(len(self.corpus))
            
        try:
            query_vec = self.vectorizer.transform([query])
        except:
            # If query contains terms not in vocabulary
            return np.zeros(len(self.corpus))
        
        scores = np.zeros(len(self.corpus))
        
        # If no terms in query match the vocabulary
        if query_vec.nnz == 0:
            return scores
            
        query_terms = query_vec.indices
        query_tf = query_vec.data
        
        for i, term in enumerate(query_terms):
            # Get precomputed values
            q_tf = query_tf[i]
            doc_tf = self.doc_term_matrix[:, term].toarray().flatten()
            idf_term = self.idf[term]
            
            # BM25 score calculation with length normalization
            doc_len_norm = 1 - self.b + self.b * (self.doc_len / self.avdl)
            term_scores = idf_term * (doc_tf * (self.k1 + 1)) / (
                doc_tf + self.k1 * doc_len_norm
            )
            
            scores += term_scores
        
        # Normalize scores to 0-100 range if requested
        if self.normalize_scores:
            scores = self._normalize_to_0_100(scores)
            
        return scores
    
    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[int, float]]:
        """
        Get top N documents most similar to query
        
        Args:
            query: The search query string
            n: Number of top results to return
            
        Returns:
            List of (document_idx, score) tuples sorted by decreasing score
            Scores will be in 0-100 range if normalize_scores is True
        """
        n = min(n, len(self.corpus))  # Ensure n doesn't exceed corpus size
        if n <= 0:
            return []
            
        scores = self.get_scores(query)
        
        # Handle case where all scores are 0
        if np.all(scores == 0):
            return [(i, 0.0) for i in range(min(n, len(scores)))]
            
        top_indices = np.argsort(scores)[::-1][:n]
        return [(idx, scores[idx]) for idx in top_indices]

def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocess text by removing special characters, extra whitespace, and optionally stopwords
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove common stopwords
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Common English stopwords (extend as needed)
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
    
    # Remove special characters and lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords if requested
    if remove_stopwords:
        words = text.split()
        text = ' '.join([word for word in words if word not in stopwords])
        
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenizer that splits text into words and removes empty tokens
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    if not isinstance(text, str):
        return []
        
    tokens = text.lower().split()
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    return tokens

def find_similar_records(df: pd.DataFrame, 
                         text_column: str, 
                         query: str, 
                         n: int = 5, 
                         preprocess: bool = True,
                         remove_stopwords: bool = False,
                         k1: float = 1.5,
                         b: float = 0.75,
                         normalize_scores: bool = True) -> List[Tuple[int, float]]:
    """
    Find records in DataFrame that are similar to the query using BM25
    
    Args:
        df: Pandas DataFrame containing records
        text_column: Column name that contains text to search in
        query: Search query
        n: Number of top results to return
        preprocess: Whether to preprocess text
        remove_stopwords: Whether to remove stopwords during preprocessing
        k1: Term frequency saturation parameter for BM25
        b: Length normalization parameter for BM25
        normalize_scores: Whether to normalize scores to 0-100 range
        
    Returns:
        List of (index, score) tuples for the top N matches
        Scores will be in 0-100 range if normalize_scores is True
    """
    # Handle empty or invalid DataFrame
    if df is None or df.empty:
        return []
        
    # Check if text_column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Extract the text column and handle NaN values
    corpus = df[text_column].fillna("").tolist()
    
    # Handle empty corpus
    if not corpus or all(not doc for doc in corpus):
        return []
    
    # Preprocess corpus and query if needed
    if preprocess:
        corpus = [preprocess_text(doc, remove_stopwords) for doc in corpus]
        query = preprocess_text(query, remove_stopwords)
    
    # Initialize BM25 with our corpus and parameters
    bm25 = BM25(corpus, tokenizer=tokenize, k1=k1, b=b, normalize_scores=normalize_scores)
    
    # Get top N similar documents
    top_matches = bm25.get_top_n(query, n=n)
    
    # Return the original DataFrame indices and scores
    return [(df.index[idx], score) for idx, score in top_matches]

# Example usage:
"""
# Create sample DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'content': [
        "Patient presents with acute abdominal pain radiating to the back",
        "Chronic back pain, history of arthritis",
        "Headache and fever for 3 days",
        "Abdominal discomfort after meals, possible GERD",
        "Patient complains of joint pain in knees and back"
    ]
})

# Search for similar records with normalized scores (0-100)
query = "CHIEF COMPLAINT: Acute abdominal and back pain"
similar_records = find_similar_records(df, 'content', query, n=3, normalize_scores=True)

# Print results
for idx, score in similar_records:
    print(f"Record #{idx} (Score: {score:.2f}/100): {df.loc[idx, 'content']}")

# Example output might be:
# Record #0 (Score: 100.00/100): Patient presents with acute abdominal pain radiating to the back
# Record #4 (Score: 65.23/100): Patient complains of joint pain in knees and back
# Record #1 (Score: 42.78/100): Chronic back pain, history of arthritis
"""