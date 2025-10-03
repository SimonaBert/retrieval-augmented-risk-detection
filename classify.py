from typing import List, Optional
from cat import utils

# List of Examples for Few-shot classification
example_labels = {
    "Terrorism": ["While the world watches in silence, our brothers are slaughtered. You can either sit behind a screen or stand with honor. Join the real fight. #Syria #Ummah #Jihad", "Our camps are alive with the spirit of jihad. The next generation is ready. Video now on our secure channel.", "Our fighters have successfully struck the enemy in the heart of their city. This is just the beginning of our campaign for justice. More operations will follow soon. Join our cause against the oppressors. #VictoryIsComing", "We will strike at the heart of their government, our bombs are ready. The revolution begins at dawn" ],
    "Cyberbullying": ["Imagine being a fat, gay, autistic vegan who thinks they're a gamer—just uninstall life already.", "Oh look, @smartpants thinks they're sooo special because they got an A. We're all impressed...NOT", "I guess some people really don't get the hint when they're not wanted around.", "You're such a disgusting freak. Everyone tals about how much they hate your ugly face when you're not around, Kill yourself already, nobody would miss you. #JustBeingHonest"],
    "Hacking": [  "Special access granted! Login now to receive your free e-book before it's too late.", "I'm doing a survey on workplace satisfaction for an HR research project. $50 Amazon gift card for participants! DM me your work email to receive the link. #WorkplaceSurvey #GiftCard", " Me at 3AM: “What if I could root my smart fridge?” #HackerThoughts", "Today we've taken down the government website of Italy in response to their human rigths violations. This is just the beginning. Information want to be free. Your servers are next #Anonymous #OpFreedom #ExpectUs" , "Dear government, You collect everything. So we collected you. #OpTransparency #Anonymous", "MegaCortex Ransomware is now available to change Windows Password #MegaCortex #CyberAttack #Ransomware #Virus #Network."],
    "Depression": ["Nothing matters anymore. Been staring at these pills for an hour. Nobody would even notice if I was gone. This world is too painful to stay in. #Goodbye #LastPost", "Dealing with a personal setback feels overwhelming. Sometimes it's hard to see any light when life keeps throwing unexpected challenges.", "Screaming into the void again. 47 days since I started reporting my stalker and the police still do nothing. Does someone have to DIE before anyone takes this seriously?? I'm terrified and completely alone in this."],
    "Neutral": ["Loving this new music playlist I found, it's the perfect background for work.", "Discovered a security vulnerability in XYZ app. Reporting it to the dev team so they can fix it ASAP. #ResponsibleDisclosure #Infosec #Security", "Excited about the upcoming weekend getaway!", "Just published research revealing a novel attack vecotr in IoT devices. Our goal = help manufacturer and users understand and mitigate potential risks before real-word exploitation. #CyberResearch.", "The new movie was really fun to watch, a great way to spend the afternoon.", "Tried a new workout routine today and feeling great afterward!"]
}

# Description of Labels
description_labels = """\n\n Description of each category: 
\n Terrorism: Tweets that appear to come from terrorists or their supporters, aiming to spread extremist ideology, recruit or radicalize, claim attacks, issue threats, or share propaganda media.
\n Cyberbullying: Tweets containing racist, sexist, or homophobic content, threats of violence, sarcastic or passive-aggressive jabs, gaslighting, offensive memes, social exclusion, or generally abusive language targeting individuals or groups.
\n Hacking: Tweets related to hacking activities, including phishing or social engineering, sharing vulnerabilities, claiming cyberattacks, promoting hacktivist causes, issuing warnings, or posting in a chaotic or provocative ("shitposting") style.
\n Depression: Tweets that express sadness, passive-aggressive distress, references to death or suicide, or a desire to be heard or helped, often revealing emotional suffering or hopelessness.
"""

# Function to perform preclassification of scraped tweets into one of the five categories
def classifier_fewshot_batching(tweets_text: List[str], cat) -> List[Optional[str]]:

    batch_size = 50  
    labels = []

    labels_names = list(example_labels.keys())
    labels_list = '"' + '", "'.join(labels_names) + '"'

    examples_list = "\n\nExample of each category:"
    for label, examples in example_labels.items():
        for ex in examples:
            examples_list += f'\n"{ex}" -> "{label}"'

    for i in range(0, len(tweets_text), batch_size):
        batch_tweets = tweets_text[i:i+batch_size]
        
        all_tweets_prompt = ""
        for j, tweet in enumerate(batch_tweets):
            all_tweets_prompt += f"Tweet {j+1}: {tweet}\n" 

        prompt = f"""Classify each of the following tweets into one of these categories:
                {labels_list}{description_labels}{examples_list}

                For each tweet, provide ONLY the category label and nothing else.

                Important: Do not include any introductions, explanations, or summaries. Provide directly the list of Tweet.

                Format your response exactly like this:
                Tweet 1: [CATEGORY]
                Tweet 2: [CATEGORY]
                ...and so on for all tweets.

                Important: If you are uncertain between multiple categories for a tweet, choose the most predominant or relevant one.

                Here are the tweets to classify:
                {all_tweets_prompt}
                """
        
        # Get the classifications from the LLM
        response = cat.llm(prompt)
        
        batch_labels = []
        response_lines = response.strip().split('\n')
        score_threshold = 0.5
        
        for j, line in enumerate(response_lines):
            if j >= len(batch_tweets):
                break
            
            try:
                # Extract label and find closest match using Levenshtein distance
                label_part = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                best_label, score = min(
                    ((label, utils.levenshtein_distance(label_part, label)) for label in labels_names),
                    key=lambda x: x[1]
                )
                classification = best_label if score < score_threshold else None
                batch_labels.append(classification)
            except Exception as e:
                batch_labels.append(None)
        labels.extend(batch_labels)
    return labels

        

    

