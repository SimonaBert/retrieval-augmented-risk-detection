import emoji
import unicodedata
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def normalize_unicode(text):
    text = unicodedata.normalize('NFKC', text) 
    return text

# Convert emojis into descriptive text
def handle_emojis(text):
    text = emoji.demojize(text)
    return text

# URL handling
def extract_url(text):
    urls =  re.findall(r'https?://[^ ]+', text)
    return urls

def handle_url(text):
    text = re.sub(r'https?://[^ ]+', '[URL]', text)
    return text

# Username handling
def extract_username(text):
    usernames =  re.findall(r'@[^ ]+', text)
    return usernames

def handle_username(text):
    text = re.sub(r'@[^ ]+', '@USERNAME', text)
    return text

# Language handling, return True if English, False otherwise
def is_english(text):
    try:
        # Remove URLs and usernames for better detection
        clean_text = re.sub(r'https?://[^ ]+', '', text)
        clean_text = re.sub(r'@[^ ]+', '', clean_text)
        # Remove repetition for better detection
        clean_text = re.sub(r'([A-Za-z])\1{2,}', r'\1', clean_text) 
        # Remove emoji descriptions for better detection 
        clean_text = re.sub(r':[a-zA-Z_]+:', '', clean_text)
        
        # Strip whitespace and check if there's enough text to analyze
        clean_text = clean_text.strip()
        
        if len(clean_text) < 3:  # Too short to reliably detect
            return False
            
        detected_lang = detect(clean_text)
        return detected_lang == 'en'
        
    except LangDetectException:
        return False

# Function to perform preprocessing
def pre_processing(text):

    if not is_english(text):
        return ""
    
    text = normalize_unicode(text)
    text = handle_emojis(text)
    text = handle_url(text)
    text = handle_username(text)

    return text