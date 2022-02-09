from apify_client import ApifyClient
import unicodedata
import pandas as pd
import os
import spacy
from tqdm import tqdm
import string

os.system('python -m spacy download en_core_web_sm')

try:
    os.mkdir('./data')
except:
    print("'data' directory already exists")

nlp = spacy.load('en_core_web_sm')


try:
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_###Input your token###")

    # Prepare the actor input
    run_input = {
        "startUrls": [{ "url": "https://www.allrecipes.com/recipes/276/desserts/cakes/" }],
        "maxItems": 5000,
        "proxyConfiguration": { "useApifyProxy": True },
    }

    # Run the actor and wait for it to finish
    run = client.actor("dtrungtin/allrecipes-scraper").call(run_input=run_input)

    # Fetch and print actor results from the run's dataset (if there are any)
    data = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        data.append(item)

    df_allR = pd.DataFrame(data)
    df_allR = df_allR[['url','name','rating','ingredients','directions']]

    df_allR.ingredients = df_allR.ingredients.apply(lambda x: x.split(', '))
    df_allR = df_allR.explode('ingredients')

    #primary cleaning
    #convert vulgar fractions 
    unicodedata.numeric(u'⅕')
    unicodedata.name(u'⅕')

    #convert vulgar fractions
    for ix, row in df_allR.iterrows():
        for char in row['ingredients']:
            if unicodedata.name(char).startswith('VULGAR FRACTION'):  
                normalized = unicodedata.normalize('NFKC', char)
                df_allR.iloc[ix, 3] = df_allR.iloc[ix, 3].replace(char, normalized)

    df_allR.to_csv('data/recepies.csv', encoding='utf-8-sig')

except:
    print('Invalid apify API token, please provide a different one!')
    

# Extracting training dataset for structured prediction
NY_train_data = pd.read_csv("https://raw.githubusercontent.com/nytimes/ingredient-phrase-tagger/master/nyt-ingredients-snapshot-2015.csv")
print("There are {0} rows in the NY Times dataset".format(NY_train_data.shape[0]))

# Lets see how many training instances we have 
print('Applying preprocessing on total: \t{} instances'.format(len(NY_train_data.name)))

# Implemting NYT data preprocessing steps

def tokenize(text):
    text = str(text)
    return [token.text for token in nlp(text) if token.text not in string.punctuation]

tqdm.pandas()

df = NY_train_data

df['input_tokenized'] = df.input.progress_apply(tokenize)
df['name_tokenized'] = df.name.progress_apply(tokenize)

# Now that we have tokenzed both our tags and our input we can move to tagging
def tagger(row):
    tokens = []
    tags = []
    for token in row['input_tokenized']:
        if token in row['name_tokenized']:
            if len(tags) == 0 or tag[-1] == 'O':
                tag = 'B - INGREDIENT'
            else:
                tag = 'I - INGREDIENT'
        else:
            tag = 'O'
            
        tokens.append(token)
        tags.append(tag)
    # Found data mistakes that need to be cleaned
    if 'B - INGREDIENT' not in tags:
        return None
    
    return list(zip(tokens, tags)) 

df['tagged_input'] = df.apply(tagger, axis = 1)
df = df.loc[df['tagged_input'].notna()]

df.to_csv('data/NER_data.csv')