import pandas as pd
from collections import Counter
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class PolicyAnalyzer:
    def __init__(self):
        self.policies = {}
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens   #(e.g., "running" -> "run") and remove stopwords
            if token.isalnum() and token not in stopwords.words('english')]
        return ' '.join(tokens)
        
    def add_policy(self, date, text, name="Facebook Platform Policy"):
        #policy version to the analyzer with preprocessed text 
        sentences = sent_tokenize(text)
        processed_text = self.preprocess_text(text)
        
        self.policies[date] = {
            'name': name,
            'text': text, #original
            'processed_text': processed_text,#cleaned
            'sentences': sentences,
            'topics': self._extract_topics(text), #topics
            'requirements': self._extract_requirements(sentences),
            'word_count': len(text.split()),
            'date': datetime.strptime(date, '%Y-%m-%d')
        }
    
    def _extract_topics(self, text):
#key text phrrases
        topics = {
            'data_collection': r'collect.*data|data.*collection|gather.*information',
            'user_consent': r'user.*consent|consent.*user|permission|authorize',
            'data_sharing': r'share.*data|data.*sharing|third.*party|partner',
            'data_protection': r'protect.*data|security|safeguard|encrypt',
            'user_rights': r'user.*right|right.*user|opt.*out|control',
            'compliance': r'comply|compliance|regulation|requirement'
        }
        
        return {topic: len(re.findall(pattern, text, re.IGNORECASE))
                for topic, pattern in topics.items()}
    
    def _extract_requirements(self, sentences):
        #extract requirement lang from policy sentences
        requirement_indicators = [
            'must', 'shall', 'required', 'need to', 'have to',
            'will not', 'cannot', 'may not'
        ]
        
        requirements = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in requirement_indicators):
                requirements.append(sentence)
                
        return requirements
    
    def calculate_semantic_similarity(self, text1, text2):
#similarity bt two texts
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    def analyze_changes(self, date1, date2):
        #compare
        if date1 not in self.policies or date2 not in self.policies:
            return "One or both dates not found"
            
        policy1 = self.policies[date1]
        policy2 = self.policies[date2]
        
        #  detect similar content regardless of where it appears in the document
        similarity = self.calculate_semantic_similarity(
            policy1['processed_text'],
            policy2['processed_text']
        )
        
        #  topic comparison
        topic_changes = {
            topic: {
                'before': policy1['topics'][topic],
                'after': policy2['topics'][topic],
                'change': policy2['topics'][topic] - policy1['topics'][topic]
            }
            for topic in policy1['topics']
        }
        
        #  requirements comparison
        old_reqs = set(policy1['requirements'])
        new_reqs = set(policy2['requirements'])
        
        # find similar requirements even if wording changed
        requirement_matches = []
        for old_req in old_reqs:
            for new_req in new_reqs:
                sim = self.calculate_semantic_similarity(
                    self.preprocess_text(old_req),
                    self.preprocess_text(new_req)
                )
                if sim > 0.8:  # threshold for similarity
                    requirement_matches.append({
                        'old': old_req,
                        'new': new_req,
                        'similarity': sim
                    })
        
        return {
            'date_range': f"{date1} to {date2}",
            'semantic_similarity': similarity,
            'word_count_change': policy2['word_count'] - policy1['word_count'],
            'topic_changes': topic_changes,
            'requirement_changes': {
                'total_before': len(old_reqs),
                'total_after': len(new_reqs),
                'similar_requirements': requirement_matches
            }
        }
    
    def generate_summary_report(self, date1, date2):
        changes = self.analyze_changes(date1, date2)
        
        report = [
            f"Policy Change Analysis: {changes['date_range']}",
            f"\nOverall Similarity: {changes['semantic_similarity']:.2%}",
            f"\nTopic Changes:"
        ]
        
        for topic, counts in changes['topic_changes'].items():
            if counts['change'] != 0:
                direction = '+' if counts['change'] > 0 else ''
                report.append(
                    f"- {topic.replace('_', ' ').title()}: {counts['before']} → "
                    f"{counts['after']} ({direction}{counts['change']})"
                )
        
        report.extend([
            f"\nRequirement Changes:",
            f"- Total requirements before: {changes['requirement_changes']['total_before']}",
            f"- Total requirements after: {changes['requirement_changes']['total_after']}",
            f"- Similar requirements found: {len(changes['requirement_changes']['similar_requirements'])}"
        ])
        
        if changes['requirement_changes']['similar_requirements']:
            report.append("\nSignificant Requirement Changes:")
            for match in changes['requirement_changes']['similar_requirements']:
                if match['similarity'] < 1.0:  # Only show changed requirements
                    report.extend([
                        f"\nOld: {match['old']}",
                        f"New: {match['new']}",
                        f"Similarity: {match['similarity']:.2%}"
                    ])
        
        return "\n".join(report)