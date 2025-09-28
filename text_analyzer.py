import nltk
import re
import ssl
import os
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st

# Handle SSL issues in cloud environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set up NLTK data path for Replit
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# Download required NLTK data with error handling
def download_nltk_resource(resource_name, resource_path):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(resource_name, download_dir=nltk_data_path, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource_name}: {e}")

# Download all required resources
download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('punkt_tab', 'tokenizers/punkt_tab')
download_nltk_resource('stopwords', 'corpora/stopwords')
download_nltk_resource('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
download_nltk_resource('averaged_perceptron_tagger_eng', 'taggers/averaged_perceptron_tagger_eng')
download_nltk_resource('wordnet', 'corpora/wordnet')

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
        Extract intelligent keyphrases from research papers using advanced NLP
        
        Args:
            texts: List of text documents
            min_freq: Minimum frequency for a keyword to be included
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of (keyphrase, frequency) tuples sorted by relevance
        """
        if not texts:
            return []
        
        # Extract both single terms and multi-word phrases
        single_terms = self._extract_technical_terms(texts)
        multi_word_phrases = self._extract_noun_phrases(texts)
        
        # Combine and score all terms
        all_terms = {}
        
        # Add single technical terms with higher weight for rare/technical words
        for term, freq in single_terms:
            if freq >= min_freq:
                # Boost score for technical-looking terms (CamelCase, numbers, etc.)
                score = freq * self._calculate_technical_score(term)
                all_terms[term] = score
        
        # Add multi-word phrases with even higher weight
        for phrase, freq in multi_word_phrases:
            if freq >= min_freq and len(phrase.split()) >= 2:
                # Multi-word phrases get higher scores as they're more specific
                score = freq * 2.5  # Boost multi-word phrases
                all_terms[phrase] = score
        
        # Sort by score and return top results
        sorted_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)
        
        # Convert scores back to frequencies for display
        return [(term, int(score)) for term, score in sorted_terms[:max_keywords]]
    
    def _extract_technical_terms(self, texts: List[str]) -> List[Tuple[str, int]]:
        """Extract technical and domain-specific terms"""
        all_terms = []
        
        # Enhanced academic and common word filters
        enhanced_stop_words = self.stop_words.union({
            'result', 'results', 'conclusion', 'conclusions', 'findings', 'finding',
            'paper', 'papers', 'study', 'studies', 'research', 'work', 'works',
            'method', 'methods', 'approach', 'approaches', 'technique', 'techniques',
            'analysis', 'analyses', 'experiment', 'experiments', 'evaluation',
            'implementation', 'development', 'application', 'applications',
            'problem', 'problems', 'solution', 'solutions', 'issue', 'issues',
            'data', 'information', 'knowledge', 'performance', 'effectiveness',
            'improvement', 'improvements', 'enhancement', 'enhancements',
            'comparison', 'comparisons', 'investigation', 'investigations'
        })
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            # Clean text but preserve technical terms
            cleaned_text = self.preprocess_text(text)
            tokens = word_tokenize(cleaned_text)
            pos_tags = pos_tag(tokens)
            
            for word, pos in pos_tags:
                if len(word) > 3 and word not in enhanced_stop_words:
                    # Focus on nouns, proper nouns, and technical adjectives
                    if pos.startswith('NN') or pos.startswith('JJ'):
                        # Give higher weight to technical-looking terms
                        if self._is_technical_term(word):
                            lemmatized = self.lemmatizer.lemmatize(word)
                            all_terms.append(lemmatized)
        
        return Counter(all_terms).most_common(200)
    
    def _extract_noun_phrases(self, texts: List[str]) -> List[Tuple[str, int]]:
        """Extract meaningful multi-word noun phrases"""
        all_phrases = []
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            # Simple noun phrase extraction using POS patterns
            cleaned_text = self.preprocess_text(text)
            tokens = word_tokenize(cleaned_text)
            pos_tags = pos_tag(tokens)
            
            # Look for patterns like "Adjective + Noun" or "Noun + Noun"
            phrases = self._extract_phrase_patterns(pos_tags)
            all_phrases.extend(phrases)
        
        # Filter phrases and return most common
        phrase_counts = Counter(all_phrases)
        
        # Filter out phrases containing too many common words
        filtered_phrases = []
        for phrase, count in phrase_counts.items():
            if self._is_meaningful_phrase(phrase):
                filtered_phrases.append((phrase, count))
        
        return sorted(filtered_phrases, key=lambda x: x[1], reverse=True)[:100]
    
    def _extract_phrase_patterns(self, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """Extract noun phrases using POS tag patterns"""
        phrases = []
        current_phrase = []
        
        for i, (word, pos) in enumerate(pos_tags):
            # Start or continue a noun phrase
            if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VBG'):
                if len(word) > 2 and word not in self.stop_words:
                    current_phrase.append(word)
            else:
                # End current phrase if it has 2+ words
                if len(current_phrase) >= 2:
                    phrase = ' '.join(current_phrase)
                    if len(phrase) > 5:  # Minimum phrase length
                        phrases.append(phrase)
                current_phrase = []
        
        # Don't forget the last phrase
        if len(current_phrase) >= 2:
            phrase = ' '.join(current_phrase)
            if len(phrase) > 5:
                phrases.append(phrase)
        
        return phrases
    
    def _is_technical_term(self, word: str) -> bool:
        """Determine if a word looks like a technical term"""
        # Check for technical indicators
        has_numbers = any(c.isdigit() for c in word)
        has_capitals = any(c.isupper() for c in word[1:])  # CamelCase
        is_long = len(word) > 6
        has_prefixes = any(word.startswith(prefix) for prefix in 
                          ['multi', 'semi', 'anti', 'pre', 'post', 'non', 'meta', 'hyper'])
        has_suffixes = any(word.endswith(suffix) for suffix in 
                          ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ology', 'ographic'])
        
        return has_numbers or has_capitals or (is_long and (has_prefixes or has_suffixes))
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase contains meaningful research concepts"""
        words = phrase.lower().split()
        
        # Skip if too many common words
        common_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        common_count = sum(1 for word in words if word in common_words)
        
        # Skip if more than 30% common words
        if len(words) > 0 and common_count / len(words) > 0.3:
            return False
        
        # Skip overly generic academic phrases
        generic_phrases = {
            'proposed method', 'new approach', 'experimental results', 'future work',
            'related work', 'previous work', 'real world', 'case study', 'data set',
            'large scale', 'high quality', 'state art', 'machine learning'
        }
        
        if phrase.lower() in generic_phrases:
            return False
        
        return True
    
    def _calculate_technical_score(self, term: str) -> float:
        """Calculate a technical relevance score for a term"""
        score = 1.0
        
        # Boost for technical indicators
        if self._is_technical_term(term):
            score *= 1.5
        
        # Boost for longer, more specific terms
        if len(term) > 8:
            score *= 1.2
        
        # Boost for terms with specific patterns
        if any(pattern in term.lower() for pattern in ['neural', 'quantum', 'algorithm', 'model', 'network']):
            score *= 1.3
        
        return score
    
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
        Cluster texts into intelligent research topics using enhanced NLP
        
        Args:
            texts: List of text documents
            n_clusters: Number of clusters/topics
            
        Returns:
            List of topic dictionaries with meaningful keyphrases and paper indices
        """
        if len(texts) < n_clusters:
            n_clusters = max(1, len(texts) // 2)
        
        if not texts or n_clusters < 1:
            return []
        
        # Enhanced preprocessing for research papers
        processed_texts = []
        for text in texts:
            # Extract both the full text and key technical terms
            cleaned = self.preprocess_text(text)
            # Focus on noun phrases and technical terms for topic modeling
            keyphrases = self._extract_text_keyphrases_for_topics(text)
            # Combine cleaned text with extracted keyphrases for richer representation
            enhanced_text = cleaned + ' ' + ' '.join(keyphrases)
            processed_texts.append(enhanced_text)
        
        # Remove empty texts
        non_empty_texts = [(i, text) for i, text in enumerate(processed_texts) if text.strip()]
        
        if len(non_empty_texts) < n_clusters:
            return []
        
        indices, clean_texts = zip(*non_empty_texts)
        
        try:
            # Enhanced TF-IDF with focus on meaningful phrases
            vectorizer = TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better phrases
                min_df=2,
                max_df=0.7,
                token_pattern=r'\b[a-zA-Z][a-zA-Z\-]+[a-zA-Z]\b',  # Include hyphenated terms
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True  # Better handling of term frequencies
            )
            
            tfidf_matrix = vectorizer.fit_transform(clean_texts)
            
            if tfidf_matrix.shape[0] < n_clusters:
                n_clusters = tfidf_matrix.shape[0]
            
            # Use K-means clustering with better initialization
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                n_init='auto',
                max_iter=300,
                init='k-means++'
            )
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            topics = []
            for i in range(n_clusters):
                # Get cluster center and top features
                cluster_center = kmeans.cluster_centers_[i]
                
                # Get top features with higher threshold for quality
                top_indices = cluster_center.argsort()[-15:][::-1]
                all_features = [feature_names[idx] for idx in top_indices]
                
                # Filter for meaningful research terms
                meaningful_keywords = self._filter_topic_keywords(all_features)
                
                # Get papers in this cluster
                cluster_papers = [indices[j] for j, label in enumerate(cluster_labels) if label == i]
                
                # Calculate topic coherence score
                coherence_score = self._calculate_topic_coherence(cluster_center, top_indices)
                
                topics.append({
                    'keywords': meaningful_keywords[:8],  # Top 8 meaningful terms
                    'papers': cluster_papers,
                    'size': len(cluster_papers),
                    'coherence': coherence_score
                })
            
            # Sort topics by size and coherence
            topics.sort(key=lambda x: (x['size'], x['coherence']), reverse=True)
            
            return topics
            
        except Exception as e:
            st.warning(f"Error in topic clustering: {str(e)}")
            return []
    
    def _extract_text_keyphrases_for_topics(self, text: str) -> List[str]:
        """Extract key phrases specifically for topic modeling"""
        if not isinstance(text, str):
            return []
        
        cleaned_text = self.preprocess_text(text)
        tokens = word_tokenize(cleaned_text)
        pos_tags = pos_tag(tokens)
        
        keyphrases = []
        current_phrase = []
        
        for word, pos in pos_tags:
            # Build noun phrases and technical terms
            if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VBG')) and \
               len(word) > 2 and word not in self.stop_words:
                current_phrase.append(word)
            else:
                if len(current_phrase) >= 1:
                    if len(current_phrase) == 1:
                        # Single technical term
                        if self._is_technical_term(current_phrase[0]):
                            keyphrases.append(current_phrase[0])
                    else:
                        # Multi-word phrase
                        phrase = ' '.join(current_phrase)
                        if self._is_meaningful_phrase(phrase):
                            keyphrases.append(phrase)
                current_phrase = []
        
        # Don't forget last phrase
        if len(current_phrase) >= 1:
            if len(current_phrase) == 1 and self._is_technical_term(current_phrase[0]):
                keyphrases.append(current_phrase[0])
            elif len(current_phrase) > 1:
                phrase = ' '.join(current_phrase)
                if self._is_meaningful_phrase(phrase):
                    keyphrases.append(phrase)
        
        return keyphrases
    
    def _filter_topic_keywords(self, keywords: List[str]) -> List[str]:
        """Filter topic keywords to focus on meaningful research terms"""
        filtered = []
        
        # Additional topic-specific filters
        topic_stop_words = {
            'show', 'shows', 'shown', 'present', 'presents', 'presented',
            'provide', 'provides', 'provided', 'demonstrate', 'demonstrates',
            'use', 'used', 'uses', 'based', 'propose', 'proposed', 'approach'
        }
        
        for keyword in keywords:
            # Skip if it's a common topic word
            if keyword.lower() in topic_stop_words:
                continue
            
            # Prioritize multi-word phrases
            if ' ' in keyword:
                if self._is_meaningful_phrase(keyword):
                    filtered.append(keyword)
            # Include technical single words
            elif len(keyword) > 4 and self._is_technical_term(keyword):
                filtered.append(keyword)
            # Include domain-specific terms
            elif any(pattern in keyword.lower() for pattern in 
                    ['neural', 'quantum', 'algorithm', 'network', 'learning', 'optimization',
                     'classification', 'regression', 'deep', 'graph', 'vision', 'language']):
                filtered.append(keyword)
        
        return filtered
    
    def _calculate_topic_coherence(self, cluster_center: np.ndarray, top_indices: np.ndarray) -> float:
        """Calculate a simple coherence score for topic quality"""
        # Simple coherence based on the concentration of top terms
        top_scores = cluster_center[top_indices]
        if len(top_scores) > 1:
            # Higher coherence if top terms have similar high scores
            return float(np.std(top_scores[-5:]))  # Coherence of top 5 terms
        return 0.0
    
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
