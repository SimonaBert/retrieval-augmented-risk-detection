from apify_client import ApifyClient
from typing import List, Dict, Union

API_KEY = "YOUR KEY"

# Function to perform scraping via APIFY actor
def scrape_profile(profile_URL: str, number_tweets: str, date: str) -> List[Dict[str, Union[int, str]]]:
    actor_identifier = "2dZb9qNraqcbL8CXP"
    client = ApifyClient(API_KEY)

    run_input = {
        "start_urls": [{"url": profile_URL}],
        "since_date": date,
        "result_count": number_tweets,
    }
    
    run = client.actor(actor_identifier).call(run_input=run_input)

    dataset = client.dataset(run["defaultDatasetId"])
    if not dataset:
        return []

    items = list(dataset.iterate_items())
    tweets = []
    tweet_id = 1
    if not items:
        return []
    else:
        for item in items:
            tweet_data = {
                "tweet_id": tweet_id,
                "full_text": item.get("full_text", ""),
                "created_at": item.get("created_at", ""),
                "label": ""
            }
            tweets.append(tweet_data)
            tweet_id += 1
    return tweets
