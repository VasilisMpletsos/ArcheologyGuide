# Load the necessary libraries
import torch
import pandas as pd
import random
import numpy as np
import warnings

from parrot import Parrot
from typing import Union
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# ---------------------- Initiallization Section ----------------------- #

# Load general descriptions
general_descriptions = pd.read_csv('./data/general_informations.csv')
general_descriptions = general_descriptions['general_informations'].to_list()
print('INFO:     Loaded General Descriptions')

# Load the paraphraser model
paraphraser = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
print('INFO:     Loaded Paraphraser Model')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
similarity_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('INFO:     Loaded Similarity Model')

# building_passages = []
# building_embeddings = []
# for i in range(0, 6):
#     building_passages.append(pd.read_csv('data/building' + str(i) + '_passages.csv')['passages'].to_list())
#     encoded_input = similarity_tokenizer(building_passages[i], padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = similarity_model(**encoded_input)
#     embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#     building_embeddings.append(embeddings.detach().numpy())

def paraphrase(text):
    # Split text into sentences
    sentences = text.split('.')
    # Keep sentences with more than 20 characters
    sentences = [sentence for sentence in sentences if len(sentence) > 20]
    # Initialize final sentences
    final_sentences = []
    # Loop through sentences
    for sentence in sentences:
        # Paraphrase sentence with chance of 70%
        if random.random() < 0.50:
            # Paraphrase sentence
            para_phrases = paraphraser.augment(input_phrase=sentence,
                                            diversity_ranker="levenshtein",
                                            do_diverse=False, 
                                            adequacy_threshold = 0.5, 
                                            fluency_threshold = 0.5)
            # If there are paraphrases generated
            if para_phrases is not None:
                # Get a random paraphrase
                random_answer = random.choice(para_phrases)[0]
                final_sentences.append(random_answer)
            else:
                # Append the original sentence
                final_sentences.append(sentence)
        else:
            # Append the original sentence
            final_sentences.append(sentence)
    # Drop randomly 25% of the sentences
    final_sentences = [sentence for sentence in final_sentences if random.random() > 0.25]
    # Shuffle final sentences
    random.shuffle(final_sentences)
    # Join sentences
    text = '. '.join(final_sentences)
    return text

# ---------------------- General Section ----------------------- #

intro = [
    'You are looking at ',
    'This is the ',
    'In front of you is the ',
    'You are standing in front of the ',
    'You have opened ',
    'You are in front of ',
    'You are looking at the ',
    'You are seeing ',
    'Now you are looking at the ',
    'Now you are looking at ',
    'Now you are looking ',
    'Now you have opened the ',
    'Now you have opened ',
]

views = {
    '1': 'building 1 of the Middle Plateau. ',
    '2': 'building 2 of the Middle Plateau. ',
    '3': 'building 3 of the Middle Plateau. ',
    '4': 'building 4 of the Middle Plateau. ',
    '5': 'building 5 of the Middle Plateau. ',
    '6': 'building 6 of the Middle Plateau. ',
}

known_views = views.keys()

# ---------------------- API Section ----------------------- #


@app.get("/view/{view_id}")
def get_building_general_informations(view_id: int):
    random_intro = random.choice(intro)
    if str(view_id) in known_views:
        answer = paraphrase(general_descriptions[int(view_id) - 1])
        answer = random_intro + views[str(view_id)] + answer
        return {"story": answer}
    else:
        return {"story": None}

@app.get("/questions/{view_id}")
def read_item(view_id: int, question: Union[str, None] = None):
    # tokenized_query = similarity_tokenizer(question, padding=True, truncation=True, return_tensors='pt')
    # embedded_query = similarity_model(**tokenized_query)
    # question_embedding = mean_pooling(embedded_query, tokenized_query['attention_mask'])
    # question_embedding = question_embedding.detach().numpy()
    # if view_id in known_buildings:
    #     building_index = known_buildings.index(view_id)
    #     similarities = cosine_similarity(question_embedding, building_embeddings[building_index])
    #     most_similar_passage_index = np.argmax(similarities)
    #     return {'passage': building_passages[building_index][most_similar_passage_index]}
    # else:
    return {'error': 'Building not found'}