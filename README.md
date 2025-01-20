**To see the results, simply run all cells in main.ipynb**

I examined shifts in Facebook's Developer Privacy Policy. I used the Wayback machine to capture the Developer Policies during  key periods of public scrutiny: 2013 (when Kogan’s “thisisyourdigitallife” was released), 2015 (when Facebook was first notified of CA’s data harvesting), March of 2018 (when public found out about the CA - Facebook case), August of 2018 (months after public disclosure), March 1st, 2019 (days before the public was notified of Cultura Collectiva/At the Pool), May of 2019 (a few months after public disclosure), and current day. Essentially, I wanted to examine Facebook’s vulnerabilities and/or improvements in their Developer Privacy Policy. 

Pre-processing: this includes cleaning the text (removing noise like special characters, punctuation), tokenization (breaking down the text into words or phrases), normalization (converting all words to lowercase, stemming, or lemmatization to reduce words to their base forms), and removing stopwords (commonly used words like "and", "the", etc.). Additionally, I used `TfidfVectorizer` ( a `scikit-learn` package) to vectorize the documents such that they were suitable for documents, as analysis algorithms work mostly with numerical data. 

Analysis: I implemented the following measurements for this text analysis:

- **Topic changes** to identify shifts in policy focus. To identify topics, `_extract_topics` defines topics by patterns of keywords and counts how many times each pattern appears, indicating the emphasis on various topics in the policy.
- **Requirement changes** to identify added, removed, or significantly altered “musts”. This identifies the how legally binding a policy is. `_extract_requirements` identifies the ‘requirements’.
- **Semantic similarity** to identify changes in nuanced language. For example,  “I enjoy jogging in the morning" and "I like running early in the day” have the same meaning but different wordings. In the context of policies that are written in formal language, subtle phrasing differences can drastically change the meaning of the policy or allow/prohibit loopholes.

