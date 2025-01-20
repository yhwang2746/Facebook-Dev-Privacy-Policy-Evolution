from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

class PolicySentimentAnalyzer:
    def __init__(self):
        # downloading required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
    def analyze_sentiment(self, text):
        #anayze sentiment of text at both document and sentence level
        blob = TextBlob(text)
        doc_sentiment = blob.sentiment.polarity
        doc_subjectivity = blob.sentiment.subjectivity
        
        # Sentence-level analysis
        sentences = sent_tokenize(text)
        sentence_sentiments = []
        
        for sentence in sentences:
            sent_blob = TextBlob(sentence)
            sentiment = sent_blob.sentiment.polarity
            subjectivity = sent_blob.sentiment.subjectivity
            sentence_sentiments.append({
                'text': sentence,
                'sentiment': sentiment,
                'subjectivity': subjectivity
            })
            
        # Find most positive and negative sentences
        sorted_sentences = sorted(sentence_sentiments, 
                                key=lambda x: x['sentiment'])
        most_negative = sorted_sentences[:3]
        most_positive = sorted_sentences[-3:]
        
        return {
            'document_sentiment': doc_sentiment,
            'document_subjectivity': doc_subjectivity,
            'average_sentence_sentiment': np.mean([s['sentiment'] 
                                                 for s in sentence_sentiments]),
            'sentiment_distribution': {
                'positive': len([s for s in sentence_sentiments 
                               if s['sentiment'] > 0]),
                'neutral': len([s for s in sentence_sentiments 
                              if s['sentiment'] == 0]),
                'negative': len([s for s in sentence_sentiments 
                               if s['sentiment'] < 0])
            },
            'most_negative_sentences': most_negative,
            'most_positive_sentences': most_positive
        }
    
    def compare_policies(self, old_text, new_text):
        #ompare sentiment between two policy versions.
        old_analysis = self.analyze_sentiment(old_text)
        new_analysis = self.analyze_sentiment(new_text)
        
        return {
            'sentiment_change': new_analysis['document_sentiment'] - 
                              old_analysis['document_sentiment'],
            'subjectivity_change': new_analysis['document_subjectivity'] - 
                                 old_analysis['document_subjectivity'],
            'avg_sentiment_change': new_analysis['average_sentence_sentiment'] - 
                                  old_analysis['average_sentence_sentiment'],
            'distribution_change': {
                'positive': new_analysis['sentiment_distribution']['positive'] - 
                           old_analysis['sentiment_distribution']['positive'],
                'neutral': new_analysis['sentiment_distribution']['neutral'] - 
                          old_analysis['sentiment_distribution']['neutral'],
                'negative': new_analysis['sentiment_distribution']['negative'] - 
                           old_analysis['sentiment_distribution']['negative']
            },
            'old_analysis': old_analysis,
            'new_analysis': new_analysis
        }

def generate_sentiment_report(comparison):
    report = [
        "Sentiment Analysis:",
        "----------------",
        f"Overall Sentiment Change: {comparison['sentiment_change']:.3f}",
        f"Subjectivity Change: {comparison['subjectivity_change']:.3f}",
        f"Average Sentence Sentiment Change: {comparison['avg_sentiment_change']:.3f}",
        "\nSentence Distribution Changes:",
        f"- Positive sentences: {comparison['distribution_change']['positive']:+d}",
        f"- Neutral sentences: {comparison['distribution_change']['neutral']:+d}",
        f"- Negative sentences: {comparison['distribution_change']['negative']:+d}",
        "\nMost Negative Statements (New Policy):"
    ]
    
    for sentence in comparison['new_analysis']['most_negative_sentences']:
        report.append(f"- {sentence['text'][:100]}... ({sentence['sentiment']:.3f})")
        
    report.extend([
        "\nMost Positive Statements (New Policy):"
    ])
    
    for sentence in comparison['new_analysis']['most_positive_sentences']:
        report.append(f"- {sentence['text'][:100]}... ({sentence['sentiment']:.3f})")
        
    return "\n".join(report)