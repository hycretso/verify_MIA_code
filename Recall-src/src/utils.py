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

os.environ['HF_ENDPOINT'] = 'hf-mirror.com'

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def prepare_dataset(data, length, num_shots):
    if data == "WikiMIA":
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
        member_data = []
        nonmember_data = []
        for data in dataset:
            if data["label"] == 1:
                member_data.append(data["input"])
            elif data["label"] == 0:
                nonmember_data.append(data["input"])
        
        # shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]

        # ✅ Limit the number of samples to 20
        num_samples = 10  # Since we will have 2 samples per iteration (one nonmember and one member)
        nonmember_data = nonmember_data[:num_samples]
        member_data = member_data[:num_samples]
        
    else: 
        raise ValueError(f"Unknown dataset: {data}. Please modify the code to include the dataset. Make sure the dataset is in the same format.")
    
    full_data = []
    # binary classification, the data need to be balanced. 
    for nm_data, m_data in zip(nonmember_data, member_data):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})

    # ✅ Debug 输出，确认数量
    print(f"Number of samples in full_data: {len(full_data)}")  # 应该输出 20

    return full_data, nonmember_prefix, member_data_prefix

def load_model(name1, name2):
    if "pythia-6.9b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    elif "pythia-2.8b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    elif "pythia-1.4b" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    elif "pythia-160m" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    elif "pythia-6.9b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    elif "pythia-2.8b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
    elif "pythia-1.4b-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
    elif "pythia-160m-dedup" == name1:
        model1 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
    elif "llama2-13b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    elif "llama2-7b" == name1:
        model1 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    elif "falcon-7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    elif "opt-6.7b" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    elif "gpt2-xl" == name1:
        model1 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    else:
        """
        You could modify the code here to make it compatible with other models
        """
        raise ValueError(f"Model {name1} is not currently supported!")

    if "pythia-6.9b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    elif "pythia-2.8b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    elif "pythia-1.4b" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    elif "pythia-160m" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m") 
    elif "pythia-6.9b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
    elif "pythia-2.8b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-2.8b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
    elif "pythia-1.4b-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
    elif "pythia-160m-dedup" == name2:
        model2 = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped") 
    elif "llama2-13b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    elif "llama2-7b" == name2:
        model2 = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    elif "falcon-7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
    elif "opt-6.7b" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
    elif "gpt2-xl" == name2:
        model2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl", return_dict=True, device_map='auto', torch_dtype=torch.float16)
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    else:
        """
        You could modify the code here to make it compatible with other models
        """
        raise ValueError(f"Model {name2} is not currently supported!")

    return model1, model2, tokenizer1, tokenizer2


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

def fitting(model, embedding_model, tokenizer, text, decoding):
    
    sementic_similarity = calculateTextSimilarity(model, embedding_model, tokenizer, text, decoding, device=model.device)
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model, tokenizer, device=model.device)

    slope, intercept = np.polyfit(np.array(sementic_similarity), np.array(all_prob), 1)

    return slope, intercept