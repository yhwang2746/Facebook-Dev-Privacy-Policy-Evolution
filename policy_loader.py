import pandas as pd
from bs4 import BeautifulSoup
import re
from pathlib import Path

class PolicyFileLoader:
    def __init__(self):
        self.raw_texts = {}
        
    def load_html_file(self, file_path, date):
        """
        Load and clean policy text from an HTML file.
        
        Parameters:
        file_path (str): Path to the HTML file
        date (str): Date of the policy in YYYY-MM-DD format
        """
        # r file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract text, removing scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        # cleaning  text
        text = self._clean_text(text)
        
        self.raw_texts[date] = text
        return text
    
    def _clean_text(self, text):
        """Clean and normalize the policy text."""
        # removing extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # removing  URLs
        text = re.sub(r'http\S+', '', text)
        
        # removing special characters; keeping periods and basic punctuation
        text = re.sub(r'[^\w\s.,;:\-\'\"()]+', ' ', text)
        
        # normalizing whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def load_multiple_files(self, file_dict):
        """
        Load multiple policy files at once.
        
        Parameters:
        file_dict (dict): Dictionary mapping dates to file paths
        Example: {'2018-01-01': 'path/to/pre_cambridge.html'}
        """
        loaded_texts = {}
        for date, path in file_dict.items():
            loaded_texts[date] = self.load_html_file(path, date)
        return loaded_texts

def create_analyzer_with_files(file_paths):
    """
    Create and initialize a PolicyAnalyzer with files.
    
    Parameters:
    file_paths (dict): Dictionary mapping dates to file paths
    
    Returns:
    PolicyAnalyzer: Initialized analyzer with loaded policies
    """
    from policy_analyzer import PolicyAnalyzer  
    
    loader = PolicyFileLoader()
    analyzer = PolicyAnalyzer()
    
    # load all files
    policy_texts = loader.load_multiple_files(file_paths)
    
    # adding each version to the analyzer
    for date, text in policy_texts.items():
        analyzer.add_policy(date, text)
        
    return analyzer