import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import streamlit as st
from typing import List, Dict, Optional
import re

class ArxivClient:
    """Client for interacting with the arXiv API"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = 3  # seconds between requests
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting to respect arXiv API guidelines"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_papers(self, category: str, max_results: int = 100, 
                     start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None) -> List[Dict]:
        """
        Search for papers in a specific category
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG')
            max_results: Maximum number of results to return
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            
        Returns:
            List of paper dictionaries
        """
        self._rate_limit()
        
        # Construct query
        query_parts = [f"cat:{category}"]
        
        # Add date range if specified
        if start_date and end_date:
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            query_parts.append(f"submittedDate:[{start_str}0000 TO {end_str}2359]")
        
        search_query = " AND ".join(query_parts)
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            
            # Additional date filtering if needed (arXiv date filtering can be imprecise)
            if start_date and end_date:
                papers = self._filter_papers_by_date(papers, start_date, end_date)
            
            return papers
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch papers from arXiv: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing arXiv response: {str(e)}")
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """Parse the XML response from arXiv API"""
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers = []
            
            for entry in root.findall('atom:entry', namespaces):
                try:
                    paper = self._parse_paper_entry(entry, namespaces)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    st.warning(f"Error parsing paper entry: {str(e)}")
                    continue
            
            return papers
            
        except ET.ParseError as e:
            raise Exception(f"Failed to parse XML response: {str(e)}")
    
    def _parse_paper_entry(self, entry, namespaces: Dict[str, str]) -> Dict:
        """Parse a single paper entry from XML"""
        paper = {}
        
        # Title
        title_elem = entry.find('atom:title', namespaces)
        paper['title'] = title_elem.text.strip() if title_elem is not None else "No title"
        
        # Abstract
        summary_elem = entry.find('atom:summary', namespaces)
        paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else "No abstract"
        
        # Authors
        authors = []
        for author in entry.findall('atom:author', namespaces):
            name_elem = author.find('atom:name', namespaces)
            if name_elem is not None:
                authors.append(name_elem.text.strip())
        paper['authors'] = authors
        
        # Published date
        published_elem = entry.find('atom:published', namespaces)
        if published_elem is not None:
            paper['published'] = published_elem.text.strip()
        
        # Updated date
        updated_elem = entry.find('atom:updated', namespaces)
        if updated_elem is not None:
            paper['updated'] = updated_elem.text.strip()
        
        # arXiv ID and URL
        id_elem = entry.find('atom:id', namespaces)
        if id_elem is not None:
            arxiv_url = id_elem.text.strip()
            paper['arxiv_url'] = arxiv_url
            # Extract arXiv ID from URL
            paper['id'] = arxiv_url.split('/')[-1]
        
        # Categories
        categories = []
        for category in entry.findall('atom:category', namespaces):
            term = category.get('term')
            if term:
                categories.append(term)
        paper['categories'] = categories
        
        # PDF link
        for link in entry.findall('atom:link', namespaces):
            if link.get('type') == 'application/pdf':
                paper['pdf_url'] = link.get('href')
                break
        
        # DOI if available
        doi_elem = entry.find('arxiv:doi', namespaces)
        if doi_elem is not None:
            paper['doi'] = doi_elem.text.strip()
        
        return paper
    
    def _filter_papers_by_date(self, papers: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Additional date filtering for papers"""
        filtered_papers = []
        
        for paper in papers:
            try:
                # Parse the published date
                published_str = paper.get('published', '')
                if published_str:
                    # arXiv dates are in ISO format: 2023-12-01T09:00:00Z
                    published_date = datetime.fromisoformat(published_str.replace('Z', '+00:00')).date()
                    
                    if start_date.date() <= published_date <= end_date.date():
                        filtered_papers.append(paper)
            except (ValueError, AttributeError) as e:
                # If date parsing fails, include the paper anyway
                filtered_papers.append(paper)
                continue
        
        return filtered_papers
    
    def get_paper_details(self, arxiv_id: str) -> Optional[Dict]:
        """Get detailed information for a specific paper by arXiv ID"""
        self._rate_limit()
        
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            return papers[0] if papers else None
            
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch paper details: {str(e)}")
    
    def search_by_author(self, author_name: str, max_results: int = 50) -> List[Dict]:
        """Search for papers by a specific author"""
        self._rate_limit()
        
        # Clean author name for search
        clean_name = re.sub(r'[^\w\s]', '', author_name)
        search_query = f"au:{clean_name}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except requests.RequestException as e:
            raise Exception(f"Failed to search papers by author: {str(e)}")
    
    def search_by_keywords(self, keywords: List[str], max_results: int = 100) -> List[Dict]:
        """Search for papers containing specific keywords in title or abstract"""
        self._rate_limit()
        
        # Construct query with keywords
        keyword_queries = []
        for keyword in keywords:
            clean_keyword = re.sub(r'[^\w\s]', '', keyword)
            keyword_queries.append(f"ti:{clean_keyword} OR abs:{clean_keyword}")
        
        search_query = " OR ".join(keyword_queries)
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except requests.RequestException as e:
            raise Exception(f"Failed to search papers by keywords: {str(e)}")
