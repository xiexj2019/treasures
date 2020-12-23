
# docs = [...]

############################################
# Step 1: use sentencepiece to remake docs
# Aims:
#    1. reduce vocab size and complex of 
#       word2vec training.
#    2. alleviate the effect of word form
############################################

import sentencepiece as spc

def encode_as_pieces(docs):
    # generate sentencepiece corpus
    with open("data/sentencepice.corpus.txt", "w") as fout:
        for doc in docs:
            fout.write(doc + "\n")
            
    # train sentence piece model
    spc.SentencePieceTrainer.Train(
        "--input=data/sentencepice.corpus.txt " + 
        "--model_prefix=data/sentencepice.trained " + 
        "--vocab_size=5000 --model_type=bpe")
    
    # load trained sentencepiece model
    sps = spc.SentencePieceProcessor()
    sps.load("data/sentencepice.trained.model")
    
    # encode raw corpus
    tmps = []
    for doc in docs:
        tmps.append(sps.encode_as_pieces(doc))
        
    return tmps, sps

docs, sps = encode_as_pieces(docs)


############################################
# Step 2: train word embedding using `docs`
# Aims: convert docs to vectors, facalitate
#       similarity caculation and clustering
############################################

with open("data/embedding.corpus.txt", "w") as fout:
    for doc in docs:
        fout.write(" ".join(doc) + "\n")
        
import os

cmd = """
./fastText/fasttext cbow \
    -input  data/embedding.corpus.txt \
    -output data/embedding \
    -minCount 0 -wordNgrams 0 -minn 0 -maxn 0 \
    -lr 0.01 -lrUpdateRate 10000 -dim 64 -ws 12 \
    -epoch 500 -neg 5 -thread 15 -seed 42
"""
os.system(cmd)


############################################
# Step 3: compute text vector using `docs`
# Aims: convert docs to vectors, facalitate
#       similarity caculation and clustering
############################################

def load_pretrained_vectors(pretrained_path):
    pretrained_vectors = {}
    
    with open(pretrained_path) as finp:
        finp.readline()
        for line in finp:
            splits = line.strip().split(" ")
            token, vector = splits[0], list(map(float, splits[1:]))
            pretrained_vectors[token] = vector
            
    return pretrained_vectors

pretrained_vectors = load_pretrained_vectors("data/embedding.vec")

import numpy as np

def compute_average_vector(tokens, pretrained_vectors, embedding_size):
    average_vector = []
    
    for token in tokens:
        if not token:
            continue
            
        try:
            average_vector.append(pretrained_vectors[token])
        except:
            print("miss match [", token, "]")
            
    return (np.array(average_vector).mean(axis=0) 
            if average_vector else np.zeros((embedding_size,)))
            
def compute_representations(docs, sps):
    reprs, index_map = [], {}

    for idx, doc in enumerate(docs):
        index_map[idx] = sps.decode(doc)

        reprs.append(
            compute_average_vector(
                doc, pretrained_vectors, 64))
        
    return np.vstack(reprs), index_map

reprs, index_map = compute_representations(docs, sps)


############################################
# Step 4: clustering and assign doc labels
# Aims: merge similar docs with thresold of
#       `eps` (the smaller, the stricter)
############################################

from sklearn.cluster import DBSCAN

cluster = DBSCAN(eps=0.01, min_samples=2, metric="cosine", n_jobs=25)
labels  = cluster.fit_predict(reprs)


############################################
# Step 5: merge similar docs with labels
# Aims: merge similar docs and normalize
############################################

import re

def strip_dirty_words(content):
    clean = re.sub("[^a-z ]+", "", content)
    # self-defined replacing rules
    # clean = (clean.replace("the", "")
    #               .replace("th ", " ")
    clean = re.sub("[ ]{2,}", " ", clean)
    
    return " ".join([word for word in clean.split(" ") 
                     if len(word) > 1])


import Levenshtein as lvs

def get_common_content(str_a, str_b):
    if len(str_a) == len(str_b):
        return str_a
    
    matching_blocks = lvs.matching_blocks(
        lvs.editops(str_a, str_b), str_a, str_b)

    content = ""
    for index, _, length in matching_blocks[:-1]:
        content += str_a[index : index + length]

    return content

import random

def normalize_cluster_title(elements):
    title = ""
    
    for _ in range(5):
        elm_str1, elm_str2 = random.choices(elements, k=2)
        if elm_str1 == elm_str2:
            continue
            
        clean = get_common_content(elm_str1, elm_str2)
        
        if len(clean) > len(title):
            title = clean
            
    return strip_dirty_words(title)
    
    
from collections import defaultdict

def gen_linking_dict(labels, index_map):
    linking_dict = defaultdict(str)
    cluster_dict = defaultdict(list)
    
    for idx, lab in enumerate(labels):
        element = index_map[idx]
        if lab == -1:
            linking_dict[element] = strip_dirty_words(element)
        else:
            cluster_dict[lab].append(index_map[idx])
        
    for lab, elements in cluster_dict.items():
        title = ""
        for element in elements:
            if len(element) > len(title):
                title = element
                
        title = normalize_cluster_title(elements)
        for element in elements:
            linking_dict[element] = title
            
    return linking_dict, cluster_dict

# linking_dict: raw doc --> normed doc
# cluster_dict: cluster label --> raw docs
linking_dict, cluster_dict = gen_linking_dict(labels, index_map)
