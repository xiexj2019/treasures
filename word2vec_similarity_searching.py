
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
            if not token.strip():
                continue
                
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
# Step 4: fit `reprs` using NearestNeighbors
# Aims: build index of `reprs` to speed up
#       target retrieval
############################################

from sklearn.neighbors import NearestNeighbors

# change n_neighbors and leaf_size according to actual situations
nsn = NearestNeighbors(n_neighbors=1, leaf_size=1, metric="cosine")

nsn.fit(reprs)


############################################
# Step 5: run similar target searching
############################################

import os

from multiprocessing import Pool

def worker(targets):
    cnt, pid = 0, os.getpid()
    with open("pros/{}.txt".format(pid), "w") as fout:
        for target in targets:
            cnt += 1
            if cnt % 100 == 0:
                print(pid, cnt)
                
            tgt = compute_average_vector(
                sps.encode_as_pieces(target),
                pretrained_vectors, 64)
                
            # distances and indices
            dst, idx = nsn.space.kneighbors([tgt])
        
        # fout.write(... + "\n")

def manager(targets):
    with Pool(20) as p:
        p.map(worker, targets)
        p.close()
        p.join()
        
# single process: run worker(targets)
# multiple process: run manager(targets)
