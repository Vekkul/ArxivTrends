import nltk
import re
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

class TextAnalyzer:
    """Text analysis utilities for research papers"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stop words
        self.stop_words.update([
            'paper', 'study', 'research', 'work', 'method', 'approach',
            'result', 'conclusion', 'analysis', 'experiment', 'show',
            'present', 'propose', 'introduce', 'describe', 'discuss',
            'demonstrate', 'investigate', 'examine', 'consider', 'provide',
            'use', 'using', 'based', 'problem', 'solution', 'application',
            'model', 'system', 'framework', 'technique', 'algorithm'
        ])
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits, keep letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords(self, texts: List[str], min_freq: int = 2, 
                        max_keywords: int = 100) -> List[Tuple[str, int]]:
        """
        Extract keywords from a list of texts using frequency analysis
        
        Args:
            texts: List of text documents
            min_freq: Minimum frequency for a keyword to be included
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of (keyword, frequency) tuples sorted by frequency
        """
        if not texts:
            return []
        
        all_keywords = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Tokenize
            tokens = word_tokenize(cleaned_text)
            
            # POS tagging to keep only nouns and adjectives
            pos_tags = pos_tag(tokens)
            
            # Filter for relevant POS tags and remove stop words
            keywords = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ')) and \
                   len(word) > 2 and \
                   word not in self.stop_words:
                    # Lemmatize the word
                    lemmatized = self.lemmatizer.lemmatize(word)
                    keywords.append(lemmatized)
            
            all_keywords.extend(keywords)
        
        # Count frequencies
        keyword_counts = Counter(all_keywords)
        
        # Filter by minimum frequency and return top keywords
        filtered_keywords = [(k, v) for k, v in keyword_counts.items() if v >= min_freq]
        filtered_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_keywords[:max_keywords]
    
    def extract_phrases(self, texts: List[str], min_freq: int = 2,
                       phrase_length: int = 2) -> List[Tuple[str, int]]:
        """Extract multi-word phrases from texts"""
        if not texts:
            return []
        
        all_phrases = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Tokenize
            tokens = word_tokenize(cleaned_text)
            
            # Remove stop words and short words
            filtered_tokens = [
                token for token in tokens 
                if len(token) > 2 and token not in self.stop_words
            ]
            
            # Extract n-grams
            for i in range(len(filtered_tokens) - phrase_length + 1):
                phrase = ' '.join(filtered_tokens[i:i + phrase_length])
                all_phrases.append(phrase)
        
        # Count frequencies
        phrase_counts = Counter(all_phrases)
        
        # Filter by minimum frequency
        filtered_phrases = [(p, c) for p, c in phrase_counts.items() if c >= min_freq]
        filtered_phrases.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_phrases[:50]  # Return top 50 phrases
    
    def cluster_topics(self, texts: List[str], n_clusters: int = 5) -> List[Dict]:
        """
        Cluster texts into topics using K-means and TF-IDF
        
        Args:
            texts: List of text documents
            n_clusters: Number of clusters/topics
            
        Returns:
            List of topic dictionaries with keywords and paper indices
        """
        if len(texts) < n_clusters:
            n_clusters = max(1, len(texts) // 2)
        
        if not texts or n_clusters < 1:
            return []
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Remove empty texts
        non_empty_texts = [(i, text) for i, text in enumerate(processed_texts) if text.strip()]
        
        if len(non_empty_texts) < n_clusters:
            return []
        
        indices, clean_texts = zip(*non_empty_texts)
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(clean_texts)
            
            if tfidf_matrix.shape[0] < n_clusters:
                n_clusters = tfidf_matrix.shape[0]
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            topics = []
            for i in range(n_clusters):
                # Get cluster center
                cluster_center = kmeans.cluster_centers_[i]
                
                # Get top keywords for this cluster
                top_indices = cluster_center.argsort()[-10:][::-1]
                keywords = [feature_names[idx] for idx in top_indices]
                
                # Get papers in this cluster
                cluster_papers = [indices[j] for j, label in enumerate(cluster_labels) if label == i]
                
                topics.append({
                    'keywords': keywords,
                    'papers': cluster_papers,
                    'size': len(cluster_papers)
                })
            
            # Sort topics by size
            topics.sort(key=lambda x: x['size'], reverse=True)
            
            return topics
            
        except Exception as e:
            st.warning(f"Error in topic clustering: {str(e)}")
            return []
    
    def analyze_sentiment_basic(self, texts: List[str]) -> Dict[str, float]:
        """
        Basic sentiment analysis using word counts
        This is a simple implementation - for production use, consider VADER or TextBlob
        """
        if not texts:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        # Simple positive and negative word lists
        positive_words = set([
            'good', 'great', 'excellent', 'outstanding', 'superior', 'effective',
            'efficient', 'successful', 'improved', 'better', 'best', 'optimal',
            'advance', 'breakthrough', 'novel', 'innovative', 'robust', 'accurate',
            'precise', 'reliable', 'stable', 'consistent', 'significant'
        ])
        
        negative_words = set([
            'bad', 'poor', 'weak', 'inadequate', 'insufficient', 'limited',
            'problematic', 'difficult', 'challenging', 'complex', 'error',
            'failure', 'incorrect', 'inaccurate', 'unreliable', 'unstable',
            'inconsistent', 'biased', 'flawed', 'wrong'
        ])
        
        total_positive = 0
        total_negative = 0
        total_words = 0
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            words = word_tokenize(self.preprocess_text(text))
            
            for word in words:
                total_words += 1
                if word in positive_words:
                    total_positive += 1
                elif word in negative_words:
                    total_negative += 1
        
        if total_words == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_ratio = total_positive / total_words
        negative_ratio = total_negative / total_words
        neutral_ratio = 1.0 - positive_ratio - negative_ratio
        
        return {
            "positive": positive_ratio,
            "negative": negative_ratio, 
            "neutral": neutral_ratio
        }
    
    def extract_technical_terms(self, texts: List[str]) -> List[Tuple[str, int]]:
        """Extract technical terms and abbreviations from texts"""
        if not texts:
            return []
        
        # Pattern for technical terms (abbreviations, acronyms)
        acronym_pattern = r'\b[A-Z]{2,}\b'
        technical_pattern = r'\b[A-Za-z]*[A-Z][A-Za-z]*[0-9]*\b'
        
        technical_terms = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            # Find acronyms
            acronyms = re.findall(acronym_pattern, text)
            technical_terms.extend(acronyms)
            
            # Find technical terms (CamelCase, mixed case)
            tech_terms = re.findall(technical_pattern, text)
            # Filter out common words
            tech_terms = [term for term in tech_terms 
                         if len(term) > 2 and term.lower() not in self.stop_words]
            technical_terms.extend(tech_terms)
        
        # Count frequencies
        term_counts = Counter(technical_terms)
        
        # Filter and return top terms
        filtered_terms = [(term, count) for term, count in term_counts.items() if count >= 2]
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_terms[:30]
    
    def summarize_abstracts(self, abstracts: List[str], max_sentences: int = 3) -> str:
        """Create a summary from multiple abstracts"""
        if not abstracts:
            return "No abstracts available for summarization."
        
        # Simple extractive summarization
        all_sentences = []
        
        for abstract in abstracts:
            if isinstance(abstract, str):
                sentences = sent_tokenize(abstract)
                all_sentences.extend(sentences)
        
        if not all_sentences:
            return "No sentences found in abstracts."
        
        # Simple frequency-based summarization
        # This is basic - for production, consider using more sophisticated methods
        word_freq = Counter()
        
        for sentence in all_sentences:
            words = word_tokenize(self.preprocess_text(sentence))
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    word_freq[word] += 1
        
        # Score sentences based on word frequencies
        sentence_scores = []
        
        for sentence in all_sentences:
            words = word_tokenize(self.preprocess_text(sentence))
            score = sum(word_freq.get(word, 0) for word in words)
            if len(words) > 0:
                score = score / len(words)  # Normalize by sentence length
            sentence_scores.append((sentence, score))
        
        # Get top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in sentence_scores[:max_sentences]]
        
        return ' '.join(top_sentences)
