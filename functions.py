import re
import torch
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer

# Prompt
INTRO_BLURB = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.")

# To be added as special tokens
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"


PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT_WITH_INPUT = """{intro}

{instruction_key}
{instruction}

{input_key}
{context}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    context="{context}",
    response_key=RESPONSE_KEY,
)

def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        print(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]

def preprocess(tokenizer, instruction_text, context_text=None):
    instruction = "Answer the following question only with the provided input. If no answer is found tell that you cannot answer based on this context. " + instruction_text
    if context_text:
        prompt_text = PROMPT_FOR_GENERATION_FORMAT_WITH_INPUT.format(instruction=instruction, context=context_text)
    else:
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)

    inputs = tokenizer(prompt_text, return_tensors="pt",)
    inputs["prompt_text"] = prompt_text
    inputs["instruction_text"] = instruction_text
    inputs["context_text"] = context_text
    return inputs

def forward(model, tokenizer, model_inputs, max_length=256):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)

    if input_ids.shape[1] == 0:
        input_ids = None
        attention_mask = None
        in_b = 1
    else:
        in_b = input_ids.shape[0]

    generated_sequence = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length
    )

    out_b = generated_sequence.shape[0]
    generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
    instruction_text = model_inputs.pop("instruction_text", None)

    return {
        "generated_sequence": generated_sequence, 
        "input_ids": input_ids,
    }


def postprocess(tokenizer, model_outputs):
    response_key_token_id = get_special_token_id(tokenizer, RESPONSE_KEY_NL)
    end_key_token_id = get_special_token_id(tokenizer, END_KEY)
    generated_sequence = model_outputs["generated_sequence"][0]
    
    # send it to cpu
    generated_sequence = generated_sequence.cpu()
    generated_sequence = generated_sequence.numpy().tolist()
    records = []

    for sequence in generated_sequence:
        decoded = None

        try:
            response_pos = sequence.index(response_key_token_id)
        except ValueError:
            print(f"Could not find response key {response_key_token_id} in: {sequence}")
            response_pos = None

        if response_pos:
            try:
                end_pos = sequence.index(end_key_token_id)
            except ValueError:
                print("Could not find end key, the output is truncated!")
                end_pos = None
            decoded = tokenizer.decode(sequence[response_pos + 1 : end_pos], skip_special_tokens=True).strip()
            
        if not decoded:
            # Otherwise we'll decode everything and use a regex to find the response and end.

            fully_decoded = tokenizer.decode(sequence)
            # The response appears after "### Response:".  The model has been trained to append "### End" at the
            # end.
            m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)
            if m:
                decoded = m.group(1).strip()
            else:
                # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                # return everything after "### Response:".
                m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                if m:
                    decoded = m.group(1).strip()
                else:
                    print(f"Failed to find response in:\n{fully_decoded}")
            
            
        rec = {"generated_text": decoded}
        records.append(rec)
    return records

def get_model_tokenizer(pretrained_model_name_or_path):
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype = torch.bfloat16,
    );
    model.resize_token_embeddings(len(tokenizer));
    return model, tokenizer

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