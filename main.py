# Load the necessary libraries
import torch
import pandas as pd
import random
import numpy as np
import warnings


from parrot import Parrot
from typing import Union
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline, T5ForConditionalGeneration, T5Tokenizer
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from sklearn.metrics.pairwise import cosine_similarity
from functions import *
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# Suppress warnings
warnings.filterwarnings("ignore")
random.seed(44)

app = FastAPI();
# app.add_middleware(HTTPSRedirectMiddleware)

# Load the paraphraser model
# gpu_available = torch.cuda.is_available()
# paraphraser = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=gpu_available)
tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
model = AutoModelForSeq2SeqLM.from_pretrained("./scripts/FineTunedParrotParaphraser")
model.to('cuda');
task_prefix = "paraphrase: ";
print('INFO:     Loaded Paraphraser Model')

model_name = "deepset/roberta-base-squad2"
qa = pipeline('question-answering', model=model_name, tokenizer=model_name)
print('INFO:     Loaded QA Model')

model_name = './scripts/FineTunedQuestionGeneration'
question_tokenizer = T5Tokenizer.from_pretrained(model_name)
question_model = T5ForConditionalGeneration.from_pretrained(model_name)
print('INFO:     Loaded Question Generation Model')

# ---------------------- Functions needed ----------------------- #

def create_question(passage):
    sentence_inputs = question_tokenizer([passage], return_tensors="pt", padding=True)
    preds = question_model.generate(
              sentence_inputs['input_ids'],
              do_sample=False, 
              max_length=256, 
              num_beams = 8,
              num_beam_groups = 4,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=4
              )
    generated_questions = question_tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    generated_questions = list(set(generated_questions))
    returned_question = random.choice(generated_questions);
    # Check if the final letter is a question mark
    if returned_question[-1] != '?':
        returned_question = returned_question.replace('.', '?')
    return returned_question

def paraphase_text(text):
    # para_phrases = paraphraser.augment(input_phrase=text, diversity_ranker="levenshtein", do_diverse=False, adequacy_threshold = 0.7, fluency_threshold = 0.7);
    # get 2 instructions from the dataset
    inputs = tokenizer([task_prefix + text], return_tensors="pt", padding=True)
    inputs = inputs.to('cuda')
    preds = model.generate(
              inputs['input_ids'],
              do_sample=False, 
              max_length=256, 
              num_beams = 8,
              num_beam_groups = 4,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=5
              )
    generated_phrases = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Remove duplicates
    generated_phrases = list(set(generated_phrases))
    if generated_phrases is not None:
        random_answer = random.choice(generated_phrases);
        return random_answer
    else:
        return text


# ---------------------- Initiallization Section ----------------------- #

# Load general descriptions
# reat txt file
with open('./data/intro_informations.txt', 'r') as f:
    intro_story = f.readlines()
intro_story = [clean_text(sentence) for sentence in intro_story]
general_informations = pd.read_csv('./data/general_informations.csv')
general_informations = general_informations['general_informations'].to_list()
print('INFO:     Loaded General Descriptions')

# Load the LLM Dolly v2 3b model with its tokenizer
LLM_model, LLM_tokenizer = get_model_tokenizer(pretrained_model_name_or_path = "./scripts/FineTunedDollyV2");
LLM_model = LLM_model.to('cuda');
print('INFO:     Loaded LLM Model')

# Load model from HuggingFace Hub
similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
similarity_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('INFO:     Loaded Similarity Model')

building1_passages = pd.read_csv('data/building1_passages.csv')['passages'].to_list()
building2_passages = pd.read_csv('data/building2_passages.csv')['passages'].to_list()
building3_passages = pd.read_csv('data/building3_passages.csv')['passages'].to_list()
building5_passages = pd.read_csv('data/building5_passages.csv')['passages'].to_list()
demeters_temple_passages = pd.read_csv('data/demeter_sanctuary_passages.csv')['passages'].to_list()
vryokastraki_passages = pd.read_csv('data/vryokastraki_passages.csv')['passages'].to_list()
christian_basilica_passages = pd.read_csv('data/christian_basilica_passages.csv')['passages'].to_list()
sanctuary_geometric_passages = pd.read_csv('data/sanctuary_geometric_passages.csv')['passages'].to_list()
fortress_passages = pd.read_csv('data/fortress_passages.csv')['passages'].to_list()
necropolis_passages = pd.read_csv('data/necropolis_passages.csv')['passages'].to_list()
findings = pd.read_csv('data/findings.csv')['passages'].to_list()

passages = [building1_passages, 
            building3_passages, 
            building5_passages, 
            demeters_temple_passages,
            ]

view_embeddings = []
for i, sentences in enumerate(passages):
    encoded_input = similarity_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = similarity_model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    view_embeddings.append(embeddings.detach().numpy())

def paraphrase(sentences, paraphrase_rate=0.25, keep_rate=0.30, shuffle_chance=0.25):
    # Drop randomly keep_rate % of the sentences
    kept_sentences = []
    discarded_sentences = []
    for sentence in sentences:
        if random.random() < keep_rate:
            kept_sentences.append(sentence)
        else:
            discarded_sentences.append(sentence)
    # Loop through sentences
    random_paraphrase = [random.random() < paraphrase_rate for i in range(len(kept_sentences))]
    random_phrases = [paraphase_text(sentence) if random_paraphrase[i] else sentence for i, sentence in enumerate(kept_sentences)]
    # Shuffle final sentences
    if random.random() < shuffle_chance:
        random.shuffle(random_phrases)
    return random_phrases, discarded_sentences

# ---------------------- General Section ----------------------- #

intro = [
    'Welcome to the Vryokastro. ',
    'Welcome to the Vryokastro, the ancient city of the island of Kythnos. ',
    'Welcome to the archeological site of Vryokastro. ',
    'You are in the archeological site of Vryokastro. ',
    'You are in the archeological site of Vryokastro, the ancient city of the island of Kythnos. ',
    'This is the archeological site of Vryokastro. ',
    'This is the archeological site of Vryokastro, the ancient city of the island of Kythnos. ',
]

openings = [
    'You are looking at ',
    'This is the ',
    'In front of you is the ',
    'In front of you, you can see the ',
    'You are standing in front of the ',
    'You are in front of ',
    'You are looking at the ',
    'You are looking ',
    'You are seeing ',
    'Now you are looking at the ',
    'Now you are looking at ',
    'Now you are looking ',
]

# Settings for each view
views_settings = [
    {"paraphrase_rate_setting": 0.2, "keep_rate_setting": 0.7, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.1, "keep_rate_setting": 0.8, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.2, "keep_rate_setting": 0.7, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.2, "keep_rate_setting": 0.7, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.25, "keep_rate_setting": 0.7, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.1, "keep_rate_setting": 0.75, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.1, "keep_rate_setting": 0.75, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.1, "keep_rate_setting": 0.75, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.1, "keep_rate_setting": 0.75, "shuffle_rate_setting": 0.9},
    {"paraphrase_rate_setting": 0.1, "keep_rate_setting": 0.75, "shuffle_rate_setting": 0.9},
]

# Specific views
views = {
    '0': 'Building 1 of the Middle Plateau (Asklipeio). ',
    '1': 'Building 2 of the Middle Plateau (Assistive to Building 1). ',
    '2': 'Building 3 of the Middle Plateau (Ancient Sanctuary). ',
    '3': 'Building 5 of the Middle Plateau. (Rectorate). ',
    '4': "Demeter's Sanctuary on the Acropolis. ",
    '5': 'Vryokastraki Island. ',
    '6': 'Cristian Basilica. ',
    '7': 'Sanctuary of the Geometric and Classical times. ',
    '8': 'Fortress in the southern sector of Acropolis. ',
    '9': 'Necropolis of the ancient city. ',
}
known_views = views.keys();

# For which buildings to allow questions
view_ids_allowed_questions = ['0', '2', '3', '4'];

views_findings = {
    '0': 'Pebbled floor (Building 1, Room A)',
    '1': 'Statue foundation (Building 1, Room C)',
    '2': 'Aprodite statue (Building 1, Outside Room C)',
    '3': 'Water tank Opening (Building 1, NE corner)',
    '4': 'Water tank Interior Findings',
    '5': 'Water Channel (Building 5)',
    '6': 'Staircase for upper floor (Building 5)',
    '7': 'Room A (Building 5)',
    '8': 'Room Z (Building 5)',
}
known_findings = views_findings.keys();




unanswerable_questions = [
    "I do not know the answer to this question.",
    "Sorry i cannot answer this question.",
    "Sadly i can't answer this question.",
    "I don't know the answer to this question.",
    "I don't know the answer to this question, sorry.",
    "I do not have the answer to this question.",
    "Can you ask me something else? I don't know the answer to this question.",
]

# ---------------------- API Section ----------------------- #


@app.get("/intro")
def get_intro():
    random_intro = random.choice(intro)
    intro_sentences, discarded_sentences = paraphrase(intro_story, paraphrase_rate=0, keep_rate=0.8, shuffle_chance=0.90)
    intro_sentences = ''.join(intro_sentences)
    intro_sentences = random_intro + intro_sentences
    return {"intro": intro_sentences}
    
@app.get("/views/{view_id}")
def get_building_intro(view_id: int):
    random_start = random.choice(openings)
    if str(view_id) in known_views:
        int_id = int(view_id);
        # Split text into sentences
        sentences = general_informations[int_id].split('.')
        # Keep sentences with more than 20 characters
        sentences = [clean_text(sentence) for sentence in sentences[:-1]]
        
        # Get the settings for this view
        paraphrase_rate = views_settings[int_id]['paraphrase_rate_setting'];
        keep_rate = views_settings[int_id]['keep_rate_setting'];
        shuffle_rate = views_settings[int_id]['shuffle_rate_setting'];

        view_sentences, discarded_sentences = paraphrase(sentences, paraphrase_rate, keep_rate, shuffle_rate)
        view_sentences = '. '.join(view_sentences)
        view_sentences = random_start + views[str(view_id)] + view_sentences
        
        if len(discarded_sentences) > 4:
            discarded_sentences = random.sample(discarded_sentences, 4)
            
        questions = []
        for context in discarded_sentences:
            questions.append(create_question(context))
        
        return {"story": view_sentences, "questions": questions, "question_responses": discarded_sentences, "questions_allowed": str(view_id) in view_ids_allowed_questions}
    else:
        return {'error': "Wrong view id."}

@app.get("/questions/{view_id}")
def get_answer_to_question(view_id: int, question: Union[str, None] = None):
    if str(view_id) in view_ids_allowed_questions:
        # get the index where the view id is in known_views
        view_index = view_ids_allowed_questions.index(str(view_id))
        tokenized_query = similarity_tokenizer(question, padding=True, truncation=True, return_tensors='pt')
        embedded_query = similarity_model(**tokenized_query)
        question_embedding = mean_pooling(embedded_query, tokenized_query['attention_mask'])
        question_embedding = question_embedding.detach().numpy()
        similarities = cosine_similarity(question_embedding, view_embeddings[view_index])
        max_score = float(similarities.max())
        context = passages[view_index][np.argmax(similarities)]
        
        # Calculate the answer of the QA model
        QA_input = {
            'question': question,
            'context': context
        }
        res = qa(QA_input)
        answer_qa = res['answer']
        
        # If relevant answer is found
        if max_score > 0.2:
            if res['score'] > 0.5:
                # If QA model is confident then return its answer
                return {'passage': context, 'answer': answer_qa,'score': max_score}
            else:
                # Else return the answer of the LLM model
                question = 'Answer the following question only with the provided input. If no answer is found tell that you cannot answer based on this context.' + question;
                pre_process_result = preprocess(LLM_tokenizer, question, context);
                model_result = forward(LLM_model, LLM_tokenizer, pre_process_result);
                final_output = postprocess(LLM_tokenizer, model_result);
                answer_llm = final_output[0]['generated_text'];
                return {'passage': context, 'answer_qa':answer_qa, 'answer_llm': answer_llm, 'answer': answer_llm,'score': max_score}
        else:
            return {'passage': context, 'answer': random.choice(unanswerable_questions), 'score': max_score}
    else:
        return {'error': "Wrong view id."}
    
@app.get("/findings/{finding_id}")
def get_finding(finding_id: int):
    if str(finding_id) in known_findings:
        int_finding_id = int(finding_id);
         # Split text into sentences
        sentences = findings[int_finding_id].split('.');
        # Keep sentences with more than 20 characters
        sentences = [clean_text(sentence) for sentence in sentences[:-1]];
        finding_sentence, _ = paraphrase(sentences, 0.3, 1.0, 0.0);
        finding_sentence = '. '.join(finding_sentence);
        return {"finding": finding_sentence}
    else:
        return {'error': "Wrong finding id."}
    
@app.get("/get_views")
def get_views():
        return views
    
@app.get("/get_findings")
def get_findings():
        return views_findings