# This implementation is adapted from Min-K% and WikiMIA: https://github.com/swj0419/detect-pretrain-code 
import os
from tqdm import tqdm
from process_data import create_dataset
import torch 
from options import Options
import numpy as np
import torch
import zlib
from eval import *
import os
import torch.nn.functional as F
from visualization import analyze_final_results
from transformers import set_seed
import torch 
import random
import openai 
from accelerate import Accelerator
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MambaForCausalLM,
)
import os
import torch
import json
import math
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from sentence_transformers.util import dot_score
from eval import *
from datasets import load_dataset
import os
import random
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from vectors import Sentence_Transformer
def calculatePerplexity(sentence, model, tokenizer, device):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def fitting(model, embedding_model, tokenizer, text, decoding):

    sementic_similarity = calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device=model.device)
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model, tokenizer, device=model.device)
    slope, intercept = np.polyfit(np.array(sementic_similarity), np.array(all_prob), 1)

    return slope, intercept
def get_conditional_calculateTextSimilarity(model, embedding_model, tokenizer, text, prefix, decoding, device):
    # 对prefix和text分别进行分词
    if prefix:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    else:
        prefix_ids = []
    
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # 合并token并放到指定设备上
    input_ids = torch.tensor(prefix_ids + text_ids).unsqueeze(0).to(device)
    
    # text部分在input_ids中的起始索引
    text_start_idx = len(prefix_ids)
    
    sementic_similarity = []
    # 从text的第一个token开始生成和计算
    for i in range(text_start_idx + 1, input_ids.size(1)):

        input_ids_processed = input_ids[0][:i].unsqueeze(0)
        
        input_ids_processed = input_ids[0][:i].unsqueeze(0)
        if decoding == "greedy":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "nuclear":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=True, max_new_tokens=1, top_k=0, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "contrastive":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), penalty_alpha=0.6, max_new_tokens=1, top_k=4, pad_token_id=tokenizer.eos_token_id)
        
        generated_embedding = embedding_model.encode(tokenizer.decode(generation[0][-1]))
        label_embedding = embedding_model.encode([tokenizer.decode(input_ids[0][i])]) 
        score = dot_score(label_embedding, generated_embedding)[0].item()
        if score <= 0:
            score = 1e-16
        sementic_similarity.append(math.log(score))
    
    return sementic_similarity
def calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device):

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        
    input_ids = input_ids.to(device)
    
    sementic_similarity = []
    for i in range(1,input_ids.size(1)):
        input_ids_processed = input_ids[0][:i].unsqueeze(0)
        if decoding == "greedy":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=False, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "nuclear":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), do_sample=True, max_new_tokens=1, top_k=0, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        elif decoding == "contrastive":
            generation = model.generate(input_ids=input_ids_processed, attention_mask=torch.ones_like(input_ids_processed), penalty_alpha=0.6, max_new_tokens=1, top_k=4, pad_token_id=tokenizer.eos_token_id)
        
        generated_embedding = embedding_model.encode(tokenizer.decode(generation[0][-1]))
        label_embedding = embedding_model.encode([tokenizer.decode(input_ids[0][i])]) 
        score = dot_score(label_embedding, generated_embedding)[0].item()
        if score <= 0:
            score = 1e-16
        sementic_similarity.append(math.log(score))
    
    return  sementic_similarity
def load_model(name1, name2, name3, use_float16=True):    
    accelerator = Accelerator()

    def load_specific_model(name, use_float16=False):
        if "mamba" in name:
            model = MambaForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype=torch.float16 if use_float16 else torch.float32,
                device_map="cuda:0"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype=torch.float16 if use_float16 else torch.float32,
                device_map="auto"
            )
        return model

    # Load the first model
    model1 = load_specific_model(name1, use_float16)
    tokenizer1 = AutoTokenizer.from_pretrained(name1)
    # Load the second model with the same float16 setting as model1
    model2 = load_specific_model(name2, use_float16)
    tokenizer2 = AutoTokenizer.from_pretrained(name2)

    model3 = load_specific_model(name3, use_float16)
    tokenizer3 = AutoTokenizer.from_pretrained(name3)

    model1.eval()
    model2.eval()
    model3.eval()

    model1, model2, model3 = accelerator.prepare(model1, model2, model3)

    return model1, model2, model3, tokenizer1, tokenizer2, tokenizer3, accelerator

def api_key_setup(key_path):
    openai.api_key = open(key_path, "r").read()

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def get_ll(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return get_all_prob(input_ids, loss, logits)

def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    input_encodings = tokenizer(input_text, return_tensors="pt")
    target_encodings = tokenizer(target_text, return_tensors="pt")
    concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
    labels = concat_ids.clone()
    labels[:, : input_encodings.input_ids.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, logits = outputs[:2]
    return get_all_prob(labels, loss, logits)

def get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()  # log-likelihood
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()

def inference(model1, model2, model3, embedding_model, tokenizer1, tokenizer2, tokenizer3, decoding, target_data, prefix, accelerator, num_shots, ex):
    pred = {}

    # unconditional log-likelihood
    ll = get_ll(target_data, model1, tokenizer1,accelerator.device)[1]

    # ReCaLL
    if int(num_shots) != 0:   
        # conditional log-likelihood with prefix     
        ll_nonmember = get_conditional_ll("".join(prefix), target_data, model1, tokenizer1, accelerator.device)[1]
        pred["recall"] = ll_nonmember / ll

    # baselines
    input_ids = torch.tensor(tokenizer1.encode(target_data)).unsqueeze(0).to(accelerator.device)
    with torch.no_grad():
        outputs = model1(input_ids, labels=input_ids)
    _, logits = outputs[:2]
    ll_ref = get_ll(target_data, model2, tokenizer2, accelerator.device)[1]

    # loss and zlib
    pred["ll"] = ll
    pred["ref"] = ll - ll_ref
    pred["zlib"] = ll / len(zlib.compress(bytes(target_data, "utf-8")))

    # For mink and mink++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink
    for ratio in [0.2]:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        pred[f"mink_{ratio}"] = np.mean(topk).item()

    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.2]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        pred[f"mink++_{ratio}"] = np.mean(topk).item()

    #********  新增recall + 相似度结合***********
    # unconditional text_similarity
    text_similarity = calculateTextSimilarity(model1, embedding_model, tokenizer1, target_data, decoding=decoding, device=model1.device)
    pred["Similarity"] = -np.mean(text_similarity).item()
    ll = pred["Similarity"]

    if int(num_shots) != 0:  
        # conditional text_similarity with prefix     
        text_similarity_nonmember = get_conditional_calculateTextSimilarity(model1, embedding_model, tokenizer1, target_data, "".join(prefix), decoding=decoding, device=model1.device)
        ll_nonmember = -np.mean(text_similarity_nonmember).item()
        pred["recall_similarity"] = ll_nonmember / ll

    #******** 新增 PETAL代码 ********************
    slope, intercept = fitting(model3, embedding_model, tokenizer3, target_data, decoding=decoding)

    text_similarity = calculateTextSimilarity(model1, embedding_model, tokenizer1, target_data, decoding=decoding, device=model1.device)

    all_prob_estimated = [i*slope + intercept for i in text_similarity]
    pred["PETAL"] = -np.mean(all_prob_estimated).item()

    #********  新增recall + PETAL结合***********

    ll = pred["PETAL"]

    if int(num_shots) != 0:  
        # conditional text_prob with prefix     
        text_similarity_nonmember = get_conditional_calculateTextSimilarity(model1, embedding_model, tokenizer1, target_data, "".join(prefix), decoding=decoding, device=model1.device)
        slope, intercept = fitting(model3, embedding_model, tokenizer3, "".join(prefix) + target_data, decoding=decoding)
        all_prob_estimated = [i*slope + intercept for i in text_similarity_nonmember]
        ll_nonmember = -np.mean(all_prob_estimated).item()
        pred["recall_PETAL"] = ll_nonmember / ll 


    ex["pred"] = pred
    return ex

def generate_prompt(example):
    return f"Generate a passage that is similar to the given text in length, domain, and style.\nGiven text:{example}\nPassage :"

def get_completion(prompt):
    message = [{"role": "user", "content": prompt}]
    responses = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=message,
        max_tokens=1024,
        temperature=1,
    )
    return responses.choices[0]["message"]["content"]

def gpt_synthetic_prefix (original_prefixes):
    # default generate synthetic prefix from non-member data
    synthetic_prefixes = []
    for original_prefix in original_prefixes:
        prompt = generate_prompt(original_prefix)
        response = get_completion(prompt)      
        synthetic_prefixes.append(response)
    return synthetic_prefixes

    
def process_prefix(target_model, prefix, avg_length, pass_window, total_shots):
    if pass_window:
        return prefix
    max_length = model1.config.max_position_embeddings if "mamba" not in target_model else 2048
    token_counts = [len(tokenizer1.encode(shot)) for shot in prefix]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    total_shots = max_shots
    return truncated_prefix

def evaluate_data(test_data, model1, model2, model3, embedding_model, tokenizer1, tokenizer2, tokenizer3, decoding, prefix, accelerator, total_shots, pass_window, synehtic_prefix):
    all_output = []
    if int(total_shots) != 0:
        avg_length = int(np.mean([len(tokenizer1.encode(ex["input"])) for ex in test_data])) 
        prefix = process_prefix(target_model, prefix, avg_length, pass_window, total_shots)
        if synehtic_prefix:
            prefix = gpt_synthetic_prefix(prefix)

    for ex in tqdm(test_data):
        new_ex = inference(model1, model2, model3, embedding_model, tokenizer1, tokenizer2, tokenizer3, decoding, ex["input"], prefix, accelerator, total_shots, ex)
        all_output.append(new_ex)
    return all_output

if __name__ == "__main__":
    fix_seed(42)
    args = Options()
    args = args.parser.parse_args()

    output_dir = args.output_dir
    dataset = args.dataset
    target_model = args.target_model
    ref_model = args.ref_model
    sub_dataset = args.sub_dataset
    num_shots = args.num_shots
    pass_window = args.pass_window
    synehtic_prefix = args.synehtic_prefix
    api_key_path = args.api_key_path
    surrogate_model = args.surrogate_model
    if synehtic_prefix and api_key_path is not None:
        api_key_setup(api_key_path)

    # process and prepare the data
    full_data, nonmember_prefix, member_data_prefix = create_dataset(dataset, sub_dataset, output_dir, num_shots)

    # load models
    model1, model2, model3, tokenizer1, tokenizer2, tokenizer3, accelerator = load_model(target_model, ref_model, surrogate_model)
    embedding_model = "all-MiniLM-L6-v2"
    # load embedding model
    embedding_model = Sentence_Transformer(embedding_model, model1.device)
    decoding = "greedy"
    # evaluate the data
    all_output = evaluate_data(full_data, model1, model2, model3, embedding_model, tokenizer1, tokenizer2, tokenizer3, decoding, nonmember_prefix, accelerator, num_shots, pass_window, synehtic_prefix)
    
    # save the results
    all_output_path = os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[1]}_{ref_model.split('/')[1]}", f"{sub_dataset}", f"{num_shots}_shot_{sub_dataset}.json")
    os.makedirs(os.path.dirname(all_output_path), exist_ok=True)
    dump_jsonl(all_output, all_output_path)
    print(f"Saved results to {all_output_path}")
    
    # evaluate the results
    fig_fpr_tpr(all_output, all_output_path)        
    
    # result visualizations to show 0 to n shot results - make sure you have these results 
    analyze_final_results(os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[1]}_{ref_model.split('/')[1]}", f"{sub_dataset}"), show_values=True)
            