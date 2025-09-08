import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import streamlit as st
from collections import Counter, defaultdict
import re

class DataProcessor:
    """Data processing utilities for research paper analysis"""
    
    def __init__(self):
        self.processed_data = None
        
    def process_papers_data(self, papers: List[Dict]) -> pd.DataFrame:
        """Convert list of paper dictionaries to processed DataFrame"""
        if not papers:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        # Ensure required columns exist
        required_columns = ['title', 'abstract', 'authors', 'published', 'categories']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Process publication dates
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        df['published_date'] = df['published'].dt.date
        df['published_year'] = df['published'].dt.year
        df['published_month'] = df['published'].dt.month
        
        # Process authors
        df['author_count'] = df['authors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['first_author'] = df['authors'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')
        
        # Process categories
        df['category_count'] = df['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['primary_category'] = df['categories'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')
        
        # Text length analysis
        df['title_length'] = df['title'].astype(str).str.len()
        df['abstract_length'] = df['abstract'].astype(str).str.len()
        df['abstract_word_count'] = df['abstract'].astype(str).str.split().str.len()
        
        # Extract arXiv ID if not present
        if 'id' not in df.columns and 'arxiv_url' in df.columns:
            df['id'] = df['arxiv_url'].str.extract(r'([0-9]{4}\.[0-9]{4,5})')
        
        self.processed_data = df
        return df
    
    def calculate_publication_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate publication trends and statistics"""
        if df.empty:
            return {}
        
        trends = {}
        
        # Daily publication counts
        daily_counts = df.groupby('published_date').size()
        trends['daily_counts'] = daily_counts.to_dict()
        trends['avg_daily_publications'] = daily_counts.mean()
        trends['max_daily_publications'] = daily_counts.max()
        trends['total_days'] = len(daily_counts)
        
        # Monthly trends
        monthly_counts = df.groupby([df['published'].dt.year, df['published'].dt.month]).size()
        trends['monthly_counts'] = monthly_counts.to_dict()
        
        # Weekly patterns
        df['day_of_week'] = df['published'].dt.day_name()
        weekly_pattern = df.groupby('day_of_week').size()
        trends['weekly_pattern'] = weekly_pattern.to_dict()
        
        # Peak publication periods
        if not daily_counts.empty:
            peak_day = daily_counts.idxmax()
            trends['peak_day'] = {
                'date': peak_day,
                'count': daily_counts.max()
            }
        
        # Growth trends
        if len(daily_counts) > 1:
            # Simple trend calculation (could be enhanced with regression)
            x = np.arange(len(daily_counts))
            y = daily_counts.values.astype(float)
            trend_slope = np.polyfit(x, y, 1)[0]
            trends['trend_slope'] = trend_slope
            trends['trend_direction'] = 'increasing' if trend_slope > 0 else 'decreasing'
        
        return trends
    
    def analyze_author_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze author collaboration patterns"""
        if df.empty:
            return {}
        
        author_stats = {}
        
        # Author frequency
        all_authors = []
        for authors in df['authors']:
            if isinstance(authors, list):
                all_authors.extend(authors)
        
        author_counts = Counter(all_authors)
        author_stats['total_unique_authors'] = len(author_counts)
        author_stats['top_authors'] = dict(author_counts.most_common(20))
        
        # Collaboration statistics
        author_stats['avg_authors_per_paper'] = df['author_count'].mean()
        author_stats['max_authors_per_paper'] = df['author_count'].max()
        author_stats['single_author_papers'] = (df['author_count'] == 1).sum()
        author_stats['collaboration_rate'] = 1 - (author_stats['single_author_papers'] / len(df))
        
        # Author distribution
        author_count_dist = df['author_count'].value_counts().sort_index()
        author_stats['author_count_distribution'] = author_count_dist.to_dict()
        
        # Prolific authors analysis
        prolific_authors = {author: count for author, count in author_counts.items() if count > 1}
        author_stats['prolific_authors'] = len(prolific_authors)
        author_stats['prolific_authors_list'] = dict(sorted(prolific_authors.items(), 
                                                           key=lambda x: x[1], reverse=True)[:10])
        
        return author_stats
    
    def analyze_category_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze subject category patterns"""
        if df.empty:
            return {}
        
        category_stats = {}
        
        # Flatten all categories
        all_categories = []
        for categories in df['categories']:
            if isinstance(categories, list):
                all_categories.extend(categories)
        
        category_counts = Counter(all_categories)
        category_stats['total_categories'] = len(category_counts)
        category_stats['category_distribution'] = dict(category_counts.most_common(20))
        
        # Primary category analysis
        primary_counts = df['primary_category'].value_counts()
        category_stats['primary_category_distribution'] = primary_counts.to_dict()
        
        # Multi-category papers
        multi_category_papers = df[df['category_count'] > 1]
        category_stats['multi_category_papers'] = len(multi_category_papers)
        category_stats['multi_category_rate'] = len(multi_category_papers) / len(df)
        category_stats['avg_categories_per_paper'] = df['category_count'].mean()
        
        # Category combinations
        if len(multi_category_papers) > 0:
            category_combinations = defaultdict(int)
            for categories in multi_category_papers['categories']:
                if isinstance(categories, list) and len(categories) > 1:
                    # Sort categories to ensure consistent combinations
                    combo = tuple(sorted(categories))
                    category_combinations[combo] += 1
            
            top_combinations = dict(sorted(category_combinations.items(), 
                                         key=lambda x: x[1], reverse=True)[:10])
            category_stats['top_category_combinations'] = {
                str(combo): count for combo, count in top_combinations.items()
            }
        
        return category_stats
    
    def calculate_text_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate text-based statistics"""
        if df.empty:
            return {}
        
        text_stats = {}
        
        # Title statistics
        text_stats['avg_title_length'] = df['title_length'].mean()
        text_stats['median_title_length'] = df['title_length'].median()
        text_stats['max_title_length'] = df['title_length'].max()
        text_stats['min_title_length'] = df['title_length'].min()
        
        # Abstract statistics
        text_stats['avg_abstract_length'] = df['abstract_length'].mean()
        text_stats['median_abstract_length'] = df['abstract_length'].median()
        text_stats['avg_abstract_word_count'] = df['abstract_word_count'].mean()
        text_stats['median_abstract_word_count'] = df['abstract_word_count'].median()
        
        # Find papers with unusually long/short abstracts
        q75 = df['abstract_word_count'].quantile(0.75)
        q25 = df['abstract_word_count'].quantile(0.25)
        iqr = q75 - q25
        
        outlier_threshold_high = q75 + 1.5 * iqr
        outlier_threshold_low = q25 - 1.5 * iqr
        
        text_stats['long_abstracts'] = (df['abstract_word_count'] > outlier_threshold_high).sum()
        text_stats['short_abstracts'] = (df['abstract_word_count'] < outlier_threshold_low).sum()
        
        return text_stats
    
    def detect_research_themes(self, df: pd.DataFrame, min_freq: int = 3) -> Dict[str, Any]:
        """Detect common research themes from titles and abstracts"""
        if df.empty:
            return {}
        
        # Extract common terms from titles
        all_title_words = []
        for title in df['title'].dropna():
            # Simple word extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
            all_title_words.extend(words)
        
        title_themes = Counter(all_title_words)
        
        # Filter common but not informative words
        stop_words = {'using', 'based', 'analysis', 'study', 'research', 
                     'approach', 'method', 'system', 'model', 'application'}
        
        filtered_title_themes = {word: count for word, count in title_themes.items() 
                               if count >= min_freq and word not in stop_words}
        
        themes = {
            'title_themes': dict(sorted(filtered_title_themes.items(), 
                                      key=lambda x: x[1], reverse=True)[:20]),
            'total_title_words': len(all_title_words),
            'unique_title_words': len(title_themes)
        }
        
        return themes
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        if df.empty:
            return {}
        
        summary = {
            'dataset_info': {
                'total_papers': len(df),
                'date_range': {
                    'start': df['published_date'].min(),
                    'end': df['published_date'].max(),
                    'span_days': (df['published_date'].max() - df['published_date'].min()).days
                } if not df['published_date'].isna().all() and df['published_date'].notna().any() else None,
                'data_completeness': {
                    'titles': df['title'].notna().sum(),
                    'abstracts': df['abstract'].notna().sum(),
                    'authors': df['authors'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum(),
                    'categories': df['categories'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
                }
            }
        }
        
        # Add other statistics
        try:
            summary['publication_trends'] = self.calculate_publication_trends(df)
            summary['author_patterns'] = self.analyze_author_patterns(df)
            summary['category_patterns'] = self.analyze_category_patterns(df)
            summary['text_statistics'] = self.calculate_text_statistics(df)
            summary['research_themes'] = self.detect_research_themes(df)
        except Exception as e:
            st.warning(f"Error calculating some statistics: {str(e)}")
        
        return summary
    
    def export_processed_data(self, df: pd.DataFrame, format: str = 'csv') -> str:
        """Export processed data in specified format"""
        if df.empty:
            return ""
        
        if format.lower() == 'csv':
            # Prepare DataFrame for CSV export
            export_df = df.copy()
            
            # Convert list columns to string representation
            list_columns = ['authors', 'categories']
            for col in list_columns:
                if col in export_df.columns:
                    export_df[col] = export_df[col].apply(
                        lambda x: '; '.join(x) if isinstance(x, list) else str(x)
                    )
            
            return export_df.to_csv(index=False)
        
        elif format.lower() == 'json':
            # Convert DataFrame to JSON
            return df.to_json(orient='records', date_format='iso', indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def filter_data(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply various filters to the dataset"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Date range filter
        if 'start_date' in filters and filters['start_date']:
            filtered_df = filtered_df[filtered_df['published_date'] >= filters['start_date']]
        
        if 'end_date' in filters and filters['end_date']:
            filtered_df = filtered_df[filtered_df['published_date'] <= filters['end_date']]
        
        # Author filter
        if 'author' in filters and filters['author']:
            author_filter = filters['author'].lower()
            filtered_df = filtered_df[
                filtered_df['authors'].apply(
                    lambda authors: any(author_filter in author.lower() 
                                      for author in authors) 
                    if isinstance(authors, list) else False
                )
            ]
        
        # Category filter
        if 'category' in filters and filters['category']:
            category_filter = filters['category']
            filtered_df = filtered_df[
                filtered_df['categories'].apply(
                    lambda categories: category_filter in categories 
                    if isinstance(categories, list) else False
                )
            ]
        
        # Keyword filter (search in title and abstract)
        if 'keyword' in filters and filters['keyword']:
            keyword = filters['keyword'].lower()
            filtered_df = filtered_df[
                (filtered_df['title'].str.lower().str.contains(keyword, na=False)) |
                (filtered_df['abstract'].str.lower().str.contains(keyword, na=False))
            ]
        
        # Author count filter
        if 'min_authors' in filters:
            filtered_df = filtered_df[filtered_df['author_count'] >= filters['min_authors']]
        
        if 'max_authors' in filters:
            filtered_df = filtered_df[filtered_df['author_count'] <= filters['max_authors']]
        
        return filtered_df
