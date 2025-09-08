import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from arxiv_client import ArxivClient
from text_analyzer import TextAnalyzer
from data_processor import DataProcessor
from visualizations import Visualizations

# Initialize session state
if 'papers_data' not in st.session_state:
    st.session_state.papers_data = pd.DataFrame()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Initialize components
@st.cache_resource
def get_components():
    arxiv_client = ArxivClient()
    text_analyzer = TextAnalyzer()
    data_processor = DataProcessor()
    visualizations = Visualizations()
    return arxiv_client, text_analyzer, data_processor, visualizations

arxiv_client, text_analyzer, data_processor, visualizations = get_components()

def analyze_trending_concepts(df):
    """Extract and analyze trending concepts from paper titles"""
    import re
    from collections import Counter
    
    if df.empty:
        return []
    
    # Extract meaningful words from titles
    all_concepts = []
    stop_words = {'using', 'based', 'analysis', 'study', 'research', 'approach', 
                 'method', 'system', 'model', 'application', 'paper', 'new', 
                 'novel', 'improved', 'enhanced', 'via', 'through', 'with', 'for'}
    
    for title in df['title'].fillna(''):
        # Extract meaningful terms (2+ characters, alphabetic)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        # Filter out common academic words
        meaningful_words = [word for word in words if word not in stop_words]
        all_concepts.extend(meaningful_words)
    
    # Count frequencies
    concept_counts = Counter(all_concepts)
    
    # Return top concepts with frequency >= 2
    trending = [{'concept': concept, 'frequency': freq} 
                for concept, freq in concept_counts.most_common(50) 
                if freq >= 2]
    
    return trending

def analyze_concept_evolution(df):
    """Analyze how research concepts evolve over time"""
    import re
    from collections import defaultdict
    
    if df.empty or len(df) < 10:
        return []
    
    # Group papers by time periods
    df_sorted = df.sort_values('date')
    
    # Get top concepts first
    trending_concepts = analyze_trending_concepts(df)
    if not trending_concepts:
        return []
    
    # Focus on top 5 most frequent concepts
    top_concepts = [item['concept'] for item in trending_concepts[:5]]
    
    # Track concept frequency over time
    concept_timeline = defaultdict(lambda: defaultdict(int))
    
    for _, row in df_sorted.iterrows():
        date = row['date']
        title = str(row['title']).lower()
        
        for concept in top_concepts:
            if concept in title:
                concept_timeline[concept][date] += 1
    
    # Convert to format suitable for plotting
    timeline_data = []
    for concept, dates in concept_timeline.items():
        for date, freq in dates.items():
            timeline_data.append({
                'concept': concept,
                'date': date,
                'frequency': freq
            })
    
    return timeline_data

def main():
    st.set_page_config(
        page_title="ArXiv Research Trends Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 ArXiv Research Trends Analyzer")
    st.markdown("Analyze research paper trends and patterns from arXiv using natural language processing and data visualization.")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        
        # Subject categories
        categories = {
            "CS - Artificial Intelligence": "cs.AI",
            "CS - Machine Learning": "cs.LG",
            "CS - Natural Language": "cs.CL",
            "CS - Computer Vision": "cs.CV",
            "Physics - General": "physics.gen-ph",
            "Mathematics - Statistics": "math.ST",
            "Quantitative Biology": "q-bio",
            "Statistics - ML": "stat.ML"
        }
        
        selected_category = st.selectbox(
            "Select Subject Category",
            options=list(categories.keys()),
            index=0
        )
        
        # Date range
        st.subheader("Date Range")
        end_date = st.date_input("End Date", value=datetime.now().date())
        days_back = st.number_input("Days to look back", min_value=7, max_value=365, value=30, step=1)
        start_date = end_date - timedelta(days=days_back)
        
        # Max results
        max_results = st.number_input("Maximum papers to fetch", min_value=10, max_value=500, value=100, step=10)
        
        # Search button
        if st.button("🔍 Fetch Papers", type="primary"):
            with st.spinner("Fetching papers from arXiv..."):
                try:
                    category_code = categories[selected_category]
                    papers = arxiv_client.search_papers(
                        category=category_code,
                        max_results=max_results,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.max.time())
                    )
                    
                    if papers:
                        st.session_state.papers_data = pd.DataFrame(papers)
                        st.session_state.search_performed = True
                        st.success(f"✅ Fetched {len(papers)} papers successfully!")
                        st.rerun()
                    else:
                        st.warning("No papers found for the selected criteria.")
                        
                except Exception as e:
                    st.error(f"Error fetching papers: {str(e)}")
        
        # Analysis options
        if not st.session_state.papers_data.empty:
            st.subheader("Analysis Options")
            
            # Keyword analysis parameters
            min_keyword_freq = st.number_input("Minimum keyword frequency", min_value=1, max_value=10, value=2, step=1)
            max_keywords = st.number_input("Maximum keywords to show", min_value=10, max_value=100, value=50, step=5)
            
            if st.button("🔬 Analyze Papers"):
                with st.spinner("Analyzing papers..."):
                    try:
                        # Perform text analysis
                        abstracts = st.session_state.papers_data['abstract'].fillna('').tolist()
                        keywords = text_analyzer.extract_keywords(abstracts, min_freq=min_keyword_freq)
                        topics = text_analyzer.cluster_topics(abstracts, n_clusters=min(5, len(abstracts)//2 if len(abstracts) > 0 else 1))
                        
                        # Store results
                        st.session_state.analysis_results = {
                            'keywords': keywords[:max_keywords],
                            'topics': topics,
                            'analysis_timestamp': datetime.now()
                        }
                        
                        st.success("✅ Analysis completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    # Main content area
    if st.session_state.papers_data.empty:
        st.info("👈 Use the sidebar to search for papers and start your analysis.")
        
        # Show example categories and what can be analyzed
        st.subheader("What you can analyze:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📈 Trends Analysis**
            - Publication frequency over time
            - Author collaboration patterns
            - Category popularity trends
            """)
        
        with col2:
            st.markdown("""
            **🔤 Text Analysis**
            - Keyword extraction and frequency
            - Topic clustering and modeling
            - Abstract sentiment analysis
            """)
        
        with col3:
            st.markdown("""
            **📊 Data Insights**
            - Statistical summaries
            - Export analysis results
            - Interactive visualizations
            """)
    
    else:
        # Display analysis results
        display_analysis_dashboard()

def display_analysis_dashboard():
    """Display the main analysis dashboard"""
    
    # Basic statistics
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", len(st.session_state.papers_data))
    
    with col2:
        unique_authors = set()
        for authors in st.session_state.papers_data['authors']:
            unique_authors.update(authors)
        st.metric("Unique Authors", len(unique_authors))
    
    with col3:
        date_range = pd.to_datetime(st.session_state.papers_data['published']).dt.date
        st.metric("Date Range", f"{date_range.min()} to {date_range.max()}")
    
    with col4:
        avg_authors = st.session_state.papers_data['authors'].apply(len).mean()
        st.metric("Avg Authors/Paper", f"{avg_authors:.1f}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🔤 Text Analysis", "📊 Detailed View", "💾 Export"])
    
    with tab1:
        display_trends_analysis()
    
    with tab2:
        display_text_analysis()
    
    with tab3:
        display_detailed_view()
    
    with tab4:
        display_export_options()

def display_trends_analysis():
    """Display trends and temporal analysis"""
    st.subheader("Research Trend Analysis")
    
    # Prepare data for visualization
    df = st.session_state.papers_data.copy()
    df['published'] = pd.to_datetime(df['published'])
    df['date'] = df['published'].dt.date
    
    # Extract trending concepts from titles over time
    st.subheader("📈 Trending Research Concepts")
    trending_concepts = analyze_trending_concepts(df)
    
    if trending_concepts:
        # Create trending concepts visualization
        concept_df = pd.DataFrame(trending_concepts)
        fig = px.bar(
            concept_df.head(15),
            x='frequency',
            y='concept',
            orientation='h',
            title='Most Frequent Research Concepts in Titles',
            labels={'frequency': 'Frequency', 'concept': 'Research Concept'},
            color='frequency',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Show concept evolution over time if we have sufficient date range
    date_range = (df['date'].max() - df['date'].min()).days
    if date_range > 7:  # More than a week of data
        st.subheader("📊 Research Focus Evolution")
        concept_timeline = analyze_concept_evolution(df)
        
        if concept_timeline:
            fig_timeline = px.line(
                concept_timeline,
                x='date',
                y='frequency',
                color='concept',
                title='Evolution of Key Research Concepts Over Time',
                labels={'date': 'Date', 'frequency': 'Frequency', 'concept': 'Concept'}
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Top authors by publication count
    st.subheader("Most Active Authors")
    author_counts = {}
    for authors in df['authors']:
        for author in authors:
            author_counts[author] = author_counts.get(author, 0) + 1
    
    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_authors:
        authors_df = pd.DataFrame(top_authors, columns=['Author', 'Papers'])
        fig = px.bar(
            authors_df,
            x='Papers',
            y='Author',
            orientation='h',
            title='Top 10 Most Active Authors'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_text_analysis():
    """Display text analysis results"""
    if st.session_state.analysis_results:
        st.subheader("🔤 Keyword Analysis")
        
        keywords = st.session_state.analysis_results.get('keywords', [])
        if keywords:
            # Keywords frequency chart
            keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
            
            fig = px.bar(
                keywords_df.head(20),
                x='Frequency',
                y='Keyword',
                orientation='h',
                title='Top 20 Most Frequent Keywords'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Keywords word cloud representation (using bar chart)
            st.subheader("Keyword Frequency Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Keywords:**")
                for i, (keyword, freq) in enumerate(keywords[:10], 1):
                    st.write(f"{i}. **{keyword}**: {freq} occurrences")
            
            with col2:
                # Show keyword statistics
                frequencies = [freq for _, freq in keywords]
                st.write("**Statistics:**")
                st.write(f"- Total unique keywords: {len(keywords)}")
                st.write(f"- Average frequency: {sum(frequencies)/len(frequencies):.1f}")
                st.write(f"- Max frequency: {max(frequencies)}")
                st.write(f"- Min frequency: {min(frequencies)}")
        
        # Topic clustering results
        topics = st.session_state.analysis_results.get('topics', [])
        if topics:
            st.subheader("📚 Topic Clusters")
            for i, topic in enumerate(topics, 1):
                with st.expander(f"Topic {i}: {', '.join(topic['keywords'][:3])}..."):
                    st.write(f"**Keywords:** {', '.join(topic['keywords'])}")
                    st.write(f"**Number of papers:** {len(topic['papers'])}")
                    if topic['papers']:
                        st.write("**Sample papers:**")
                        for paper_idx in topic['papers'][:3]:
                            if paper_idx < len(st.session_state.papers_data):
                                paper = st.session_state.papers_data.iloc[paper_idx]
                                st.write(f"- {paper['title']}")
    else:
        st.info("👈 Click 'Analyze Papers' in the sidebar to perform text analysis.")

def display_detailed_view():
    """Display detailed paper information"""
    st.subheader("📊 Papers Database")
    
    # Search and filter options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search in titles and abstracts")
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Publication Date", "Title", "Author Count"])
    
    # Filter data based on search
    df = st.session_state.papers_data.copy()
    
    if search_term:
        mask = (
            df['title'].str.contains(search_term, case=False, na=False) |
            df['abstract'].str.contains(search_term, case=False, na=False)
        )
        df = df[mask]
    
    # Sort data
    if sort_by == "Publication Date":
        df = df.sort_values('published', ascending=False)
    elif sort_by == "Title":
        df = df.sort_values('title')
    elif sort_by == "Author Count":
        df['author_count'] = df['authors'].apply(len)
        df = df.sort_values('author_count', ascending=False)
    
    st.write(f"Showing {len(df)} papers:")
    
    # Display papers
    for idx, paper in df.iterrows():
        with st.expander(f"📄 {paper['title'][:80]}{'...' if len(paper['title']) > 80 else ''}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Published:** {pd.to_datetime(paper['published']).strftime('%Y-%m-%d')}")
                st.write(f"**Categories:** {', '.join(paper['categories'])}")
                if paper.get('arxiv_url'):
                    st.write(f"**ArXiv Link:** [View Paper]({paper['arxiv_url']})")
            
            with col2:
                st.write(f"**Paper ID:** {paper.get('id', 'N/A')}")
                st.write(f"**Authors Count:** {len(paper['authors'])}")
            
            st.write("**Abstract:**")
            st.write(paper['abstract'])

def display_export_options():
    """Display export functionality"""
    st.subheader("💾 Export Analysis Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Export Options:**")
        
        # Export papers data
        if st.button("📊 Export Papers Data (CSV)"):
            csv_data = st.session_state.papers_data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Papers CSV",
                data=csv_data,
                file_name=f"arxiv_papers_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Export analysis results
        if st.session_state.analysis_results and st.button("🔬 Export Analysis Results (JSON)"):
            analysis_data = st.session_state.analysis_results.copy()
            analysis_data['analysis_timestamp'] = analysis_data['analysis_timestamp'].isoformat()
            json_data = json.dumps(analysis_data, indent=2)
            st.download_button(
                label="⬇️ Download Analysis JSON",
                data=json_data,
                file_name=f"arxiv_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**Summary Report:**")
        if not st.session_state.papers_data.empty:
            report = generate_summary_report()
            st.text_area("Report", value=report, height=300)
            
            st.download_button(
                label="⬇️ Download Summary Report",
                data=report,
                file_name=f"arxiv_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

def generate_summary_report():
    """Generate a text summary report"""
    df = st.session_state.papers_data
    
    report = f"""ArXiv Research Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW
================
Total Papers: {len(df)}
Date Range: {pd.to_datetime(df['published']).dt.date.min()} to {pd.to_datetime(df['published']).dt.date.max()}
Unique Authors: {len(set().union(*df['authors']))}
Average Authors per Paper: {df['authors'].apply(len).mean():.1f}

PUBLICATION TRENDS
==================
"""
    
    # Add daily publication stats
    daily_counts = df.groupby(pd.to_datetime(df['published']).dt.date).size()
    report += f"Most Active Day: {daily_counts.idxmax()} ({daily_counts.max()} papers)\n"
    report += f"Average Papers per Day: {daily_counts.mean():.1f}\n\n"
    
    # Add top authors
    author_counts = {}
    for authors in df['authors']:
        for author in authors:
            author_counts[author] = author_counts.get(author, 0) + 1
    
    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    report += "TOP AUTHORS\n===========\n"
    for i, (author, count) in enumerate(top_authors, 1):
        report += f"{i}. {author}: {count} papers\n"
    
    # Add keyword analysis if available
    if st.session_state.analysis_results:
        keywords = st.session_state.analysis_results.get('keywords', [])
        if keywords:
            report += "\nTOP KEYWORDS\n============\n"
            for i, (keyword, freq) in enumerate(keywords[:10], 1):
                report += f"{i}. {keyword}: {freq} occurrences\n"
    
    return report

if __name__ == "__main__":
    main()
