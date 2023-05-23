# Load the necessary libraries
# import torch
import pandas as pd
import numpy as np

from typing import Union
from fastapi import FastAPI
# from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
# from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# # Load model from HuggingFace Hub
# similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# similarity_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

building_passages = []
building_embeddings = []
for i in range(0, 6):
    building_passages.append(pd.read_csv('data/building' + str(i) + '_passages.csv')['passages'].to_list())
    encoded_input = similarity_tokenizer(building_passages[i], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = similarity_model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    building_embeddings.append(embeddings.detach().numpy())

@app.get("/building/{building_id}")
def get_building_general_informations(building_id: int):
    building1_story = '''This is the story of building 1, it was used as a sancuary for the people of the city ok kythnos. It is told that god Dimitra was worshiped there.'''
    building2_story = '''This is the story of building 2. It is the place where the people of the temple where living.'''
    if building_id == 1:
        return {"story": building1_story}
    elif building_id == 2:
        return {"story": building2_story}
    else:
        return {"story": None}
    
known_buildings = [1, 2, 3, 4, 5, 6]

@app.get("/questions/{building_id}")
def read_item(building_id: int, question: Union[str, None] = None):
    tokenized_query = similarity_tokenizer(question, padding=True, truncation=True, return_tensors='pt')
    embedded_query = similarity_model(**tokenized_query)
    question_embedding = mean_pooling(embedded_query, tokenized_query['attention_mask'])
    question_embedding = question_embedding.detach().numpy()
    if building_id in known_buildings:
        building_index = known_buildings.index(building_id)
        similarities = cosine_similarity(question_embedding, building_embeddings[building_index])
        most_similar_passage_index = np.argmax(similarities)
        return {'passage': building_passages[building_index][most_similar_passage_index]}
    else:
        return {'error': 'Building not found'}