import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import streamlit as st

class Visualizations:
    """Create interactive visualizations for research paper analysis"""
    
    def __init__(self):
        # Default color schemes
        self.color_palette = px.colors.qualitative.Set3
        self.sequential_colors = px.colors.sequential.Viridis
        
    def create_publication_timeline(self, df: pd.DataFrame, title: str = "Publication Timeline") -> go.Figure:
        """Create an interactive timeline of publications"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Group by date
        daily_counts = df.groupby('published_date').size().reset_index(name='count')
        daily_counts['published_date'] = pd.to_datetime(daily_counts['published_date'])
        
        fig = px.line(daily_counts, x='published_date', y='count',
                     title=title,
                     labels={'published_date': 'Publication Date', 'count': 'Number of Papers'})
        
        fig.update_traces(mode='lines+markers', hovertemplate='<b>%{x}</b><br>Papers: %{y}<extra></extra>')
        fig.update_layout(
            xaxis_title="Publication Date",
            yaxis_title="Number of Papers",
            hovermode='x unified'
        )
        
        return fig
    
    def create_category_distribution(self, df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """Create a bar chart of category distribution"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Flatten all categories
        all_categories = []
        for categories in df['categories']:
            if isinstance(categories, list):
                all_categories.extend(categories)
        
        category_counts = pd.Series(all_categories).value_counts().head(top_n)
        
        fig = px.bar(x=category_counts.values, y=category_counts.index, 
                    orientation='h',
                    title=f"Top {top_n} Subject Categories",
                    labels={'x': 'Number of Papers', 'y': 'Category'})
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, len(category_counts) * 25)
        )
        
        return fig
    
    def create_author_collaboration_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a histogram of author counts per paper"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        author_counts = df['author_count'].value_counts().sort_index()
        
        fig = px.bar(x=author_counts.index, y=author_counts.values,
                    title="Distribution of Authors per Paper",
                    labels={'x': 'Number of Authors', 'y': 'Number of Papers'})
        
        fig.update_layout(
            xaxis_title="Number of Authors",
            yaxis_title="Number of Papers",
            bargap=0.1
        )
        
        return fig
    
    def create_top_authors_chart(self, df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """Create a chart of most prolific authors"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Count author appearances
        all_authors = []
        for authors in df['authors']:
            if isinstance(authors, list):
                all_authors.extend(authors)
        
        author_counts = pd.Series(all_authors).value_counts().head(top_n)
        
        if author_counts.empty:
            fig = go.Figure()
            fig.add_annotation(text="No author data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = px.bar(x=author_counts.values, y=author_counts.index,
                    orientation='h',
                    title=f"Top {top_n} Most Active Authors",
                    labels={'x': 'Number of Papers', 'y': 'Author'})
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, len(author_counts) * 25)
        )
        
        return fig
    
    def create_keyword_frequency_chart(self, keywords: List[Tuple[str, int]], top_n: int = 20) -> go.Figure:
        """Create a horizontal bar chart of keyword frequencies"""
        if not keywords:
            fig = go.Figure()
            fig.add_annotation(text="No keyword data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Take top N keywords
        top_keywords = keywords[:top_n]
        words, frequencies = zip(*top_keywords)
        
        fig = px.bar(x=frequencies, y=words, orientation='h',
                    title=f"Top {top_n} Keywords",
                    labels={'x': 'Frequency', 'y': 'Keywords'})
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=max(400, len(top_keywords) * 25)
        )
        
        return fig
    
    def create_weekly_pattern_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a chart showing publication patterns by day of week"""
        if df.empty or 'published' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extract day of week
        df_temp = df.copy()
        df_temp['day_of_week'] = pd.to_datetime(df_temp['published']).dt.day_name()
        
        # Order days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_counts = df_temp.groupby('day_of_week').size()
        
        # Reindex to ensure correct order
        weekly_counts = weekly_counts.reindex(day_order, fill_value=0)
        
        fig = px.bar(x=weekly_counts.index, y=weekly_counts.values,
                    title="Publication Pattern by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Number of Papers'})
        
        fig.update_layout(xaxis_tickangle=-45)
        
        return fig
    
    def create_monthly_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing publication activity by month and year"""
        if df.empty or 'published' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No temporal data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Extract year and month
        df_temp = df.copy()
        df_temp['published'] = pd.to_datetime(df_temp['published'])
        df_temp['year'] = df_temp['published'].dt.year
        df_temp['month'] = df_temp['published'].dt.month
        
        # Create pivot table
        monthly_counts = df_temp.groupby(['year', 'month']).size().reset_index(name='count')
        pivot_table = monthly_counts.pivot(index='month', columns='year', values='count').fillna(0)
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=[month_names[i-1] for i in pivot_table.index],
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Year: %{x}<br>Month: %{y}<br>Papers: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Publication Heatmap by Month and Year",
            xaxis_title="Year",
            yaxis_title="Month"
        )
        
        return fig
    
    def create_abstract_length_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create a histogram of abstract lengths"""
        if df.empty or 'abstract_word_count' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No abstract length data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = px.histogram(df, x='abstract_word_count', nbins=30,
                          title="Distribution of Abstract Lengths (Word Count)",
                          labels={'abstract_word_count': 'Words in Abstract', 'count': 'Number of Papers'})
        
        fig.update_layout(
            xaxis_title="Words in Abstract",
            yaxis_title="Number of Papers"
        )
        
        return fig
    
    def create_topic_cluster_visualization(self, topics: List[Dict], max_topics: int = 10) -> go.Figure:
        """Create a visualization of topic clusters"""
        if not topics:
            fig = go.Figure()
            fig.add_annotation(text="No topic data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Prepare data for visualization
        topic_names = []
        topic_sizes = []
        topic_keywords = []
        
        for i, topic in enumerate(topics[:max_topics]):
            topic_names.append(f"Topic {i+1}")
            topic_sizes.append(topic['size'])
            topic_keywords.append(', '.join(topic['keywords'][:5]))
        
        # Create bubble chart
        fig = go.Figure(data=go.Scatter(
            x=list(range(len(topic_names))),
            y=topic_sizes,
            mode='markers',
            marker=dict(
                size=topic_sizes,
                sizemode='diameter',
                sizeref=2.*max(topic_sizes)/(40.**2),
                sizemin=4,
                color=topic_sizes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Papers in Topic")
            ),
            text=topic_keywords,
            hovertemplate='<b>%{text}</b><br>Papers: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Topic Clusters (Size = Number of Papers)",
            xaxis_title="Topics",
            yaxis_title="Number of Papers",
            xaxis=dict(tickmode='array', tickvals=list(range(len(topic_names))), ticktext=topic_names),
            height=500
        )
        
        return fig
    
    def create_collaboration_network_chart(self, df: pd.DataFrame, min_papers: int = 2) -> go.Figure:
        """Create a simple network visualization of author collaborations"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No collaboration data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Find prolific authors
        all_authors = []
        for authors in df['authors']:
            if isinstance(authors, list):
                all_authors.extend(authors)
        
        author_counts = pd.Series(all_authors).value_counts()
        prolific_authors = author_counts[author_counts >= min_papers].index.tolist()
        
        if len(prolific_authors) < 2:
            fig = go.Figure()
            fig.add_annotation(text=f"Not enough prolific authors (min {min_papers} papers)", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create collaboration matrix
        collaboration_pairs = {}
        for authors in df['authors']:
            if isinstance(authors, list) and len(authors) > 1:
                prolific_in_paper = [a for a in authors if a in prolific_authors]
                if len(prolific_in_paper) > 1:
                    for i, author1 in enumerate(prolific_in_paper):
                        for author2 in prolific_in_paper[i+1:]:
                            pair = tuple(sorted([author1, author2]))
                            collaboration_pairs[pair] = collaboration_pairs.get(pair, 0) + 1
        
        if not collaboration_pairs:
            fig = go.Figure()
            fig.add_annotation(text="No collaborations found among prolific authors", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create a simple scatter plot showing collaboration strength
        authors_list = list(set().union(*collaboration_pairs.keys()))
        author_positions = {author: i for i, author in enumerate(authors_list)}
        
        # Prepare data for scatter plot
        x_vals = []
        y_vals = []
        sizes = []
        hover_texts = []
        
        for (author1, author2), count in collaboration_pairs.items():
            x_vals.append(author_positions[author1])
            y_vals.append(author_positions[author2])
            sizes.append(count * 10)  # Scale for visibility
            hover_texts.append(f"{author1} & {author2}<br>Collaborations: {count}")
        
        fig = go.Figure(data=go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=sizes,
                sizemin=10,
                color=sizes,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Collaborations")
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Author Collaboration Network (Min {min_papers} papers each)",
            xaxis=dict(tickmode='array', tickvals=list(range(len(authors_list))), 
                      ticktext=[a[:20] + '...' if len(a) > 20 else a for a in authors_list],
                      tickangle=45),
            yaxis=dict(tickmode='array', tickvals=list(range(len(authors_list))), 
                      ticktext=[a[:20] + '...' if len(a) > 20 else a for a in authors_list]),
            height=max(500, len(authors_list) * 30)
        )
        
        return fig
    
    def create_summary_metrics_chart(self, summary_stats: Dict[str, Any]) -> go.Figure:
        """Create a dashboard-style metrics visualization"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Publication Trends", "Author Statistics", 
                          "Category Distribution", "Text Statistics"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Add placeholder data if summary_stats is empty
        if not summary_stats:
            fig.add_annotation(text="No summary statistics available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        return fig
