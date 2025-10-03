from cat.mad_hatter.decorators import hook
from datetime import datetime
from .scraping import scrape_profile
from .pre_processing import extract_url, extract_username
from .pre_processing import pre_processing
from .classify import classifier_fewshot_batching

from .table import build_distribution_table
from .table import build_tweet_table

## Hook used to disable episodic memory recall, 
## ensuring that no episodic memories are considered.
@hook  
def before_cat_recalls_episodic_memories(episodic_recall_config, cat):
    episodic_recall_config["k"] = 0
    return episodic_recall_config

## Hook used to build custom LLM chains.
## In this hook, several key steps of the pipeline are executed: scraping tweets, preprocessing, 
## performing a preliminary classification to determine the majority category in the profile, 
## filtering and reordering tweets for further analysis.  
## The processed output is then passed to the Embedder as if it were the original RAG query.
@hook
def fast_reply (_, cat):

    output = None
    
    ##### ------------------------------------------------------------------------------------------
    ##### Scraping of Tweets
    ##### ------------------------------------------------------------------------------------------

    ### Solution 1: scrape Tweets, with LLM usage for input handling
    #
    message_original = cat.working_memory.user_message_json.text
    DEFAULT_NUMBER_TWEETS = "100"
    DEFAULT_DATE = "2020-01-01"
    
    message = cat.llm(f"""
              Extract from the following text: profile URL, number of tweets, and since date.
    
              Text to analyze: {message_original}
    
              Rules: 
                - Date format must be YYYY-MM-DD
                - Output format: URL|number|date
    
              Output only the structured result, nothing else.
              """)
    
    parts = [item.strip() for item in message.split("|")]
    profile_URL = parts[0]
    number_tweets = DEFAULT_NUMBER_TWEETS
    if len(parts) > 1 and int(parts[1]) < int(DEFAULT_NUMBER_TWEETS):
       number_tweets = parts[1]
    date = parts[2] if len(parts) > 2 else DEFAULT_DATE
    cat.working_memory.profile_URL = profile_URL
    cat.working_memory.number_tweets = number_tweets
    cat.working_memory.since_date = date
    cat.working_memory.today_date = datetime.now().date()
    
    cat.send_ws_message(f'Scraping {number_tweets} tweets since date {date} from profile {profile_URL}')
    tweets = scrape_profile(profile_URL, number_tweets, date)

    ### Solution 2: scrape Tweets, without LLM usage for input handling  
    #   
    # message = cat.working_memory.user_message_json.text
    # DEFAULT_NUMBER_TWEETS = "100"
    # DEFAULT_DATE = "2020-01-01"
    #     
    # parts = [item.strip() for item in message.split("|")]
    # profile_URL = parts[0]
    # number_tweets = DEFAULT_NUMBER_TWEETS
    # if len(parts) > 1 and int(parts[1]) < int(DEFAULT_NUMBER_TWEETS):
    #      number_tweets = parts[1]
    # date = parts[2] if len(parts) > 2 else DEFAULT_DATE
    # cat.working_memory.profile_URL = profile_URL
    # cat.working_memory.number_tweets = number_tweets
    # cat.working_memory.since_date = date
    # cat.working_memory.today_date = datetime.now().date()
    #
    # cat.send_ws_message(f'Scraping {number_tweets} tweets since date {date} from profile {profile_URL}')
    # tweets = scrape_profile(profile_URL, number_tweets, date)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    ### Solution 3: use dataset
    #
    ## If you don’t want to scrape data directly from Twitter profiles, 
    ## but instead wish to run the pipeline on an existing dataset, 
    ## comment out the previous code block and uncomment the lines below.  
    ## Replace the fields with the desired values for the final report, 
    ## along with the list of tweets to be analyzed.
    
    # cat.working_memory.profile_URL = "https://twitter.com/username"
    # cat.working_memory.number_tweets = "number of tweets"
    # cat.working_memory.since_date = "since date"
    # cat.working_memory.today_date = datetime.now().date()
    #
    # tweets = [
    #     {"tweet_id": 1, "full_text": "Full text of the tweet 🔝.", "created_at": "Date in the form: 2024-05-15T16:22:00Z", "label": ""},
    #     {...}
    # ]

    if not tweets:
        result = "No tweets were found for this profile with the specified criteria."
        output = {
            "output": result
        }
        return output
    
    ##### ------------------------------------------------------------------------------------------
    ##### Preprocessing of Tweets
    ##### ------------------------------------------------------------------------------------------
    for tweet in tweets:
        tweet['url'] = extract_url(tweet['full_text']) # For final table rapresentation
        tweet['username'] = extract_username(tweet['full_text']) # For final table rapresentation
        tweet['full_text_processed'] = pre_processing(tweet['full_text'])

    # Filter only English tweets (those with non-empty full_text_processed)
    english_tweets = [tweet for tweet in tweets if tweet.get('full_text_processed', '')]
    
    # Set default label for non-English tweets
    for tweet in tweets:
        if not tweet.get('full_text_processed', ''):
            tweet['label'] = 'Not Analyzed'

    if not english_tweets:
        result = "No English tweets were found for this profile with the specified criteria."
        output = {
            "output": result
        }
        return output
    
    ##### ------------------------------------------------------------------------------------------
    ##### Preclassification of Tweets
    ##### ------------------------------------------------------------------------------------------
    full_texts = [tweet["full_text_processed"] for tweet in english_tweets]

    cat.send_ws_message("Classifing tweets.")
    labels = classifier_fewshot_batching(full_texts, cat)
    cat.working_memory.labels = labels

    for tweet, label in zip(english_tweets, labels):
        tweet["label"] = label.lower() if label is not None else "Not Classified"
    
    # Final table
    tweet_table_rows = build_tweet_table(tweets) 
    distribution_table_rows = build_distribution_table(english_tweets) 
    cat.working_memory.tweet_table_rows = tweet_table_rows
    cat.working_memory.distribution_table_rows = distribution_table_rows
   
    # Majority voting
    label_votes = {}
    for tweet in english_tweets:
        label = tweet["label"]
        if label and label.lower() != "neutral" and label.lower() != "not classified":
            label_votes[label] = label_votes.get(label, 0) + 1
                    
    if label_votes:
        max_votes = max(label_votes.values())

        majority_label = [label for label, votes in label_votes.items() if votes == max_votes][0] 
        cat.working_memory.majority_label = majority_label

        non_neutral_tweets = [
            {
                "tweet_id": tweet["tweet_id"], 
                "full_text_processed": tweet["full_text_processed"],
                "label": tweet.get("label"),
                "created_at": tweet.get("created_at")
            } 
            for tweet in english_tweets 
            if tweet.get("label") and tweet.get("label").lower() != "neutral" and tweet.get("label").lower() != "not classified"
        ]

        def get_timestamp(created_at_str):
            return datetime.fromisoformat(created_at_str.replace('Z', '+00:00')).timestamp()
            
        # Scoring function for tweets selection for further RAG analysis 
        def calculate_priority_score(tweet):
            score = 0
            
            # Classification component
            if tweet["label"] == majority_label:
                score += 100
            
            # Recency component
            if tweet.get("created_at"):
                timestamps = [get_timestamp(t.get("created_at", "")) for t in non_neutral_tweets]
                max_timestamp = max(timestamps) if timestamps else 0
                if max_timestamp > 0:
                    tweet_timestamp = get_timestamp(tweet["created_at"])
                    recency_score = (tweet_timestamp / max_timestamp) * 20
                    score += recency_score
            
            return score
        
        prioritized_tweets = sorted(non_neutral_tweets, key=calculate_priority_score, reverse=True)
        max_tweets = 10 # Maximum number of tweet further analyzed 
        selected_tweets = prioritized_tweets[:max_tweets]

        # RAG input query creation 
        all_tweets_prompt = ""
        for tweet in selected_tweets:
            all_tweets_prompt += f"Tweet ID {tweet['tweet_id']}: {tweet['full_text_processed']}\n"
        
        cat.working_memory.user_message_json.text = all_tweets_prompt

    else:
        result = "No critical tweets were found with valid labels (excluding neutral)."  
        output = {
            "output": result
        }

    return output

