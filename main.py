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

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI();

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

def clean_text(text):
    # strip sentenece
    text = text.strip()
    # remove tabs
    text = text.replace('\t', '')
    # remove new lines
    text = text.replace('\n', '')
    return text

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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


# Load the LLM model
LLM = pipeline(
    model="databricks/dolly-v2-3b", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    return_full_text=True,
    device_map="auto",
    task="text-generation"
)
# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=LLM)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
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
sanctuary_geometric_classical_times_passages = pd.read_csv('data/sanctuary_geometric_classical_times_passages.csv')['passages'].to_list()
fortress_passages = pd.read_csv('data/fortress_passages.csv')['passages'].to_list()
necropolis_passages = pd.read_csv('data/necropolis_passages.csv')['passages'].to_list()

passages = [building1_passages, 
            building2_passages, 
            building3_passages, 
            building5_passages, 
            demeters_temple_passages, 
            vryokastraki_passages, 
            christian_basilica_passages,
            sanctuary_geometric_classical_times_passages,
            fortress_passages,
            necropolis_passages]

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
    'You are standing in front of the ',
    'You have opened ',
    'You have opened the ',
    'You are in front of ',
    'You are looking at the ',
    'You are looking ',
    'You are seeing ',
    'Now you are looking at the ',
    'Now you are looking at ',
    'Now you are looking ',
    'Now you have opened the ',
    'Now you have opened ',
]

views = {
    '0': 'Building 1 of the Middle Plateau. ',
    '1': 'Building 2 of the Middle Plateau. ',
    '2': 'Building 3 of the Middle Plateau. ',
    '3': 'Building 5 of the Middle Plateau. ',
    '4': "Demeter's Sanctuary on the Acropolis. ",
    '5': 'Vryokastraki Island. ',
    '6': 'Cristian Basilica. ',
    '7': 'Sanctuary of the Geometric and Classical times. ',
    '8': 'Fortress in the southern sector of Acropolis. ',
    '9': 'Necropolis of the ancient city. ',
}
known_views = views.keys()

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
    
@app.get("/view/{view_id}")
def get_building_intro(view_id: int):
    random_start = random.choice(openings)
    if str(view_id) in known_views:
        # Split text into sentences
        sentences = general_informations[int(view_id)].split('.')
        # Keep sentences with more than 20 characters
        sentences = [clean_text(sentence) for sentence in sentences[:-1]]
        view_sentences, discarded_sentences = paraphrase(sentences, paraphrase_rate=0.1, keep_rate=0.75, shuffle_chance=0.85)
        view_sentences = '. '.join(view_sentences)
        view_sentences = random_start + views[str(view_id)] + view_sentences
        
        questions_context = discarded_sentences;
        if len(questions_context) > 4:
            questions_context = random.sample(questions_context, 4)
            
        questions = []
        for context in questions_context:
            questions.append(create_question(context))
        
        return {"story": view_sentences, "questions": questions}
    else:
        return {"story": None}

@app.get("/questions/{view_id}")
def get_answer_to_question(view_id: int, question: Union[str, None] = None):
    if str(view_id) in known_views:
        tokenized_query = similarity_tokenizer(question, padding=True, truncation=True, return_tensors='pt')
        embedded_query = similarity_model(**tokenized_query)
        question_embedding = mean_pooling(embedded_query, tokenized_query['attention_mask'])
        question_embedding = question_embedding.detach().numpy()
        similarities = cosine_similarity(question_embedding, view_embeddings[view_id])
        max_score = float(similarities.max())
        context = passages[view_id][np.argmax(similarities)]
        
        QA_input = {
            'question': question,
            'context': context
        }
        res = qa(QA_input)
        answer_qa = res['answer']
        
        question = 'Answer the following question only with the provided input. If not answer is found tell that you cannot answer based on this context.' + question;
        answer_llm = llm_context_chain.predict(instruction=question, context=context).lstrip()
        
        if max_score > 0.15 and max_score < 0.5:
            return {'passage': context, 'answer_qa':answer_qa, 'answer_llm': answer_llm, 'given_answer': answer_llm,'score': max_score}
        elif max_score >= 0.5:
            return {'passage': context, 'answer_qa':answer_qa, 'answer_llm': answer_llm, 'given_answer': answer_qa,'score': max_score}
        else:
            return {'passage': context, 'answer': random.choice(unanswerable_questions), 'score': max_score}
    else:
        return {'error': "Wrong view id."}
    
@app.get("/get_views")
def get_views():
        return views