import os

import jsonlines
import random
import numpy as np
from scipy.linalg import special_matrices
# Mock the missing triu function
special_matrices.triu = np.triu

from gensim import corpora
from gensim.summarization import bm25
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from tqdm import tqdm


class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


def bm25_preprocess(train, test, output_path, number):
    code = [' '.join(obj['code_tokens']) for obj in train]
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    
    processed = []
    id = 0
    for obj in tqdm(test, total=len(test)):
        query = obj['code_tokens']
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number]
        code_candidates_tokens = []
        for i in range(len(rtn)):
            code_candidates_tokens.append({'code_tokens': train[rtn[i][0]]['code_tokens'], 'docstring_tokens': train[rtn[i][0]]['docstring_tokens'], 'score': rtn[i][1], 'idx':i+1})            
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
        if id >= 1000:
            break
        id = id + 1
        
    with jsonlines.open(os.path.join(output_path, 'test_bm25_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)

def oracle_preprocess(train, test, output_path, number):
    code = [' '.join(obj['docstring_tokens']) for obj in train]
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    
    processed = []
    for obj in tqdm(test, total=len(test)):
        query = obj['docstring_tokens']
        score = bm25_model.get_scores(query,average_idf)
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number]
        code_candidates_tokens = []
        for i in range(len(rtn)):
            code_candidates_tokens.append({'code_tokens': train[rtn[i][0]]['code_tokens'], 'docstring_tokens': train[rtn[i][0]]['docstring_tokens'], 'score': rtn[i][1], 'idx':i+1})  
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
        
    with jsonlines.open(os.path.join(output_path, 'test_oracle_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)

def sbert_preprocess(train, test, model_path, output_path, number):
    code = [' '.join(obj['code_tokens']) for obj in train]
    other = [' '.join(obj['docstring_tokens']) for obj in train]
    
    model = SentenceTransformer(model_path)
    code_emb = model.encode(code, convert_to_tensor=True)
    processed = []
    for obj in tqdm(test):
        query = ' '.join(obj['code_tokens'])
        query_emb = model.encode(query , convert_to_tensor=True)
        hits = util.semantic_search(query_emb, code_emb, top_k=number)[0]
        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': code[hits[i]['corpus_id']].split(),'docstring_tokens': other[hits[i]['corpus_id']].split(), 'score': hits[i]['score'], 'idx':i+1})
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
    
    with jsonlines.open(os.path.join(output_path, 'test_sbert_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)

def unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    device = torch.device("cuda:1")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens=tokenizer.tokenize(' '.join(obj['code_tokens']))[:256-4]
            tokens=[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb,0)
    
    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens=tokenizer.tokenize(' '.join(obj['code_tokens']))[:256-4]
            tokens=[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings=model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb,0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score':cos_sim[i], 'corpus_id':i})
        
        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'], 'score': float(hits[i]['score']), 'idx':i+1})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'], 'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, 'test_'+str(model_name)+'_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)

def openLLM_preprocess(train, test, model_path, output_path, number):

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 使用EOS时,向右填充

    model = AutoModel.from_pretrained(model_path)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def tokens_get_hidden_states(model, inputs):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            return hidden_states

    # Process train data in batches
    code_emb = []
    batch_size = 100  # Set batch size
    for i in tqdm(range(0, len(train), batch_size)):
        batch_train = train[i:i + batch_size]
        train_texts = [' '.join(obj['code_tokens']) for obj in batch_train]
        train_inputs = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        train_hidden_states = tokens_get_hidden_states(model, train_inputs)

        train_batch_size = train_inputs['input_ids'].shape[0]
        train_last_non_padding_index = train_inputs['attention_mask'].sum(dim=1) - 1
        train_last_token_hidden_states = train_hidden_states[torch.arange(train_batch_size), train_last_non_padding_index, :].cpu().numpy()
        code_emb.append(train_last_token_hidden_states)
    code_emb = np.concatenate(code_emb, axis=0)

    del train_hidden_states, train_last_token_hidden_states
    torch.cuda.empty_cache()

    # Process test data in batches
    test_emb = []
    for i in tqdm(range(0, len(test), batch_size)):
        batch_test = test[i:i + batch_size]
        test_texts = [' '.join(obj['code_tokens']) for obj in batch_test]
        test_inputs = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        test_hidden_states = tokens_get_hidden_states(model, test_inputs)

        test_batch_size = test_inputs['input_ids'].shape[0]
        test_last_non_padding_index = test_inputs['attention_mask'].sum(dim=1) - 1
        test_last_token_hidden_states = test_hidden_states[torch.arange(test_batch_size), test_last_non_padding_index, :].cpu().numpy()
        test_emb.append(test_last_token_hidden_states)
    test_emb = np.concatenate(test_emb, axis=0)

    del test_hidden_states, test_last_token_hidden_states
    torch.cuda.empty_cache()

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score': cos_sim[i], 'corpus_id': i})
        
        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({
                'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],
                'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'],
                'score': float(hits[i]['score']),
                'idx': i+1
            })
        obj = test[idx]
        processed.append({
            'code_tokens': obj['code_tokens'],
            'docstring_tokens': obj['docstring_tokens'],
            'code_candidates_tokens': code_candidates_tokens
        })

    model_name = 'Qwen2.5-Coder-7B-Instruct' if 'Qwen2.5-Coder-7B-Instruct' in model_path else 'unknown'
    output_file = os.path.join(output_path, f'test_{model_name}_{number}.jsonl')
    with jsonlines.open(output_file, 'w') as f:
        f.write_all(processed)

train = []
with jsonlines.open('/newdisk/public/wws/Dataset/CodeSearchNet/dataset/java/train.jsonl') as f:
    for i in f:
        train.append(i)
test = []
with jsonlines.open('/newdisk/public/wws/Dataset/CodeSearchNet/dataset/java/test.jsonl') as f:
    for i in f:
        test.append(i)

# oracle_preprocess(train, test, './preprocess', 64)
# bm25_preprocess(train, test, './construction/preprocess', 5)

# sbert_preprocess(train, test, '/data/szgao/resource/pretrain/flax-sentence-embeddings/st-codesearch-distilroberta-base', './preprocess', 64) #https://huggingface.co/flax-sentence-embeddings/st-codesearch-distilroberta-base
# unixcoder_cocosoda_preprocess(train, test, 'ICL4code/models/unixcoder-base', './construction/preprocess', 64) #https://huggingface.co/microsoft/unixcoder-base
# unixcoder_cocosoda_preprocess(train, test, '/data/szgao/resource/pretrain/microsoft/cocosoda', './preprocess', 64) #https://huggingface.co/DeepSoftwareAnalytics/CoCoSoDa

# openLLM_preprocess(train, test, '/newdisk/public/wws/model_dir/Qwen2.5-Coder/Qwen2.5-Coder-7B-Instruct',
#                     '/newdisk/public/wws/ICL4code/construction/preprocess', 5)

unixcoder_cocosoda_preprocess(train, test, './models/unixcoder-base', './construction/preprocess', 5)
