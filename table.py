# Function to construct Tweets table
def build_tweet_table(all_tweets_with_labels):
    table_rows = ""
    for tweet in all_tweets_with_labels:
        tweet_id = tweet['tweet_id']
        text = tweet['full_text'].replace('|', '\\|') 
        date = tweet['created_at']
        label = tweet['label']

        urls = tweet['url']
        if isinstance(urls, list):
            url_str = "<br>".join(urls) if urls else 'N/A'
        else:
            url_str = urls
        
        usernames = tweet['username']
        if isinstance(usernames, list):
            username_str = "<br>".join(usernames) if usernames else 'N/A'
        else:
            username_str = usernames

        table_rows += f"| {tweet_id} | {text} | {date} | {url_str} | {username_str} | {label} |\n"
    return table_rows

# Function to construct frequency table
def build_distribution_table(all_tweets_with_labels):
    from collections import Counter
    
    labeled_tweets = []
    for tweet in all_tweets_with_labels:
        label = tweet['label']

        if label and label.lower() != "not classified":
            labeled_tweets.append(label)
    
    if not labeled_tweets:
        return "| No labels found | 0 | 0% |\n"

    label_counts = Counter(labeled_tweets)
    total_tweets = len(labeled_tweets)  
    
    table_rows = ""
    for label, count in label_counts.most_common():
        percentage = round((count / total_tweets) * 100, 2)
        escaped_label = label.replace('|', '\\|')
        table_rows += f"| {escaped_label} | {count} | {percentage}% |\n"
    
    return table_rows