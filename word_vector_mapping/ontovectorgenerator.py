import json, re, sys, os, collections, csv, math, time
import numpy as np
import gtbtokenize
import networkx as nx
import sklearn.metrics as metrics
from operator import itemgetter
import pandas as pd
from utils import MatrixIO, FileUtils


first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
f = lambda x: re.sub(r'[^a-z0-9]', "", x)
tokenize = lambda x: gtbtokenize.tokenize(x).lower()
EMBEDDING_SIZE = 100


N = 24358723
UNMAPPED_IDF_CONST = 4
MIN_ED = 4
vector_file = "../lod_query/biomed_vectors_p.txt"
vocab_file = "../lod_query/biomed_vocab_p.txt"
idf_file = "../lod_query/idf_file.tsv"
vocab_dict = {}
enc_vocab_dict = {}
stopWords = set([
    "a", "also", "although", "am", "an", "and", "are", ".", "NNNN", "VVVV",
    "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "being", "bill", "both",
    "bottom", "but", "by", "call", "can", "con",
    "could", "de", "do", "done", "eg", "etc", "even", "ever", 
    "find", "for", "found", "from", "get", "give", "go",
    "had", "has", "have", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "however", "if", "in", "inc", 
    "into", "is", "it", "its", "itself", "keep", "may", "me", "mine", "my", "myself", "name", "namely", "of", "onto", "our",
    "ours", "ourselves", "please", "put", "should", "show", "such", "take", "that", "the", "their", "them",
    "themselves", "these", "they", "this", "those", "though",
    "thru", "to", "us", "via", "was", "we", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "whither",
    "who", "whoever", "whom", "whose", "why", "will", "would", "yet", "you", "your", "yours", "yourself", "yourselves"])

def parse_camel_case(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

def parse_uri_token(token):
    prop = parse_camel_case(token)
    pparts = re.split("[-_]", prop)
    name = " ".join([x.title() for x in pparts])
    return name

def gettermmap_edit(term):
    return None, None

def gen_embedding(string_phrase):
    norm_attr = tokenize(string_phrase)
    attr_parts = norm_attr.strip().split()
    attr_data = []
    unmapped = []
    for part in attr_parts:
        # high prominence to exact match, then naive match and finally check edit distance
        found_match = False
        #print part
        if part in vocab_dict: attr_data.append((vocab_dict[part]["idf"], word_vecs.loc[vocab_dict[part]["index"]]))
        elif f(part) in enc_vocab_dict: 
            idf_val = vocab_dict[enc_vocab_dict[f(part)]["term"]]["idf"]
            attr_data.append((idf_val, word_vecs.loc[vocab_dict[enc_vocab_dict[f(part)]["term"]]["index"]]))
        else:
            #the gettermmap_edit is not implemented, and would return None
            attr_data.append((vocab_dict["<unk>"]["idf"], word_vecs.loc[vocab_dict["<unk>"]["index"]]))
            unmapped.append(part)
    # (idf_1*emb1 + idf_2*emb2)/(idf_1 + idf_2) 
    idf_sum = float(sum([k[0] for k in attr_data]))
    if idf_sum > 0: embedding = np.array(sum([k[0]*k[1] for k in attr_data])/idf_sum)
    else: embedding = np.zeros(EMBEDDING_SIZE)
    return embedding, unmapped

def gen_onto_embedding(onto, comp, is_arr=False):
    all_unmapped = {}
    onto_embeddings = []
    ccount = 0
    start = time.time()
    for k in onto:
        ov_embedding = []
        ov_unmapped = []
        for m in k[comp]:
            embedding, unmapped = gen_embedding(m)
            ov_embedding.append(embedding)
            if len(unmapped) > 0: ov_unmapped.append(unmapped)
        if len(ov_unmapped) > 0: all_unmapped[k["termIri"]] = ov_unmapped
        if len(k[comp]) == 0: ov_embedding = [np.zeros(EMBEDDING_SIZE)] # for empty arrays, else it prints NAN
        embedding = np.mean(np.array(ov_embedding), axis=0)
        onto_embeddings.append(embedding)
        ccount += 1
        if ccount % 10000 == 0: 
            time_elapsed = time.time()-start
            print time_elapsed, ccount
    print time_elapsed, ccount
    return np.array(onto_embeddings), all_unmapped

def get_cosine_sims(onto_embeddings, embedding):
    cosine_scores = (np.dot(onto_embeddings, embedding)/(np.linalg.norm(embedding)*np.linalg.norm(onto_embeddings, axis=1)))
    return cosine_scores

def get_cosine_sims_sk(onto_embeddings, embedding):
    cosine_scores = metrics.pairwise.cosine_similarity(onto_embeddings, embedding.reshape(1, onto_embeddings.shape[1]))
    cosine_scores = cosine_scores.flatten()
    return cosine_scores

def get_euclidean_sk(onto_embeddings, embedding):
    euclidean_scores = metrics.pairwise.euclidean_distances(onto_embeddings, embedding.reshape(1, onto_embeddings.shape[1]))
    euclidean_scores = euclidean_scores.flatten()
    return euclidean_scores

def get_top(scores, onto, comp, descending=True, count=10):
    # use descending for similarity, ascending for distance, return top 10
    sor_scores = np.argsort(scores)
    if descending: ranks = sor_scores[::-1][:count]
    else: ranks = sor_scores[0:count]
    vals = list(np.array(onto)[ranks])
    top_scores = list(scores[ranks])
    output = []
    for k in range(len(vals)): output.append((vals[k]["termIri"], vals[k][comp], top_scores[k]))
    return output

def generate_onto_vectors(onto_file):
    with open(onto_file) as fa: onto = json.load(fa)
    onto_embeddings, all_unmapped = gen_onto_embedding(onto, "skosPrefLabel")
    print onto_file, len(onto), onto_embeddings.shape, len(all_unmapped)
    return onto_embeddings, all_unmapped

def load_word_vectors(vector_file):
    df = pd.read_table(vector_file, sep=" ", header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    word_vecs = df.loc[:,1:]
    print "Loaded word vectors"
    return word_vecs

def load_vocab(vocab_file):
    vocab = open(vocab_file)
    vocab_lines = vocab.readlines()
    vocab.close()
    for k in range(len(vocab_lines)):
        vocab_parts = vocab_lines[k].strip().split()
        term = str(vocab_parts[0])
        enc_term = f(term)
        vocab_dict[term] = {"index": k, "tf": math.log(int(vocab_parts[1])), "enc": enc_term}
        if not term in enc_vocab_dict:
            enc_vocab_dict[enc_term] = {"index": k, "term": term}

    vocab_dict["<unk>"] = {"index": len(vocab_dict), "tf": 1}
    print "Loaded vocabulary"

def load_idfs(idf_file):
    idfs = open(idf_file)
    idf_lines = idfs.readlines()
    idfs.close()
    for k in range(len(idf_lines)):
        if k == 0: continue
        idf_parts = idf_lines[k].strip().split()
        term = str(idf_parts[0])
        vocab_dict[term]["idf"] = float(idf_parts[1])
    vocab_dict["<unk>"]["idf"] = UNMAPPED_IDF_CONST

word_vecs = load_word_vectors(vector_file)
load_vocab(vocab_file)
load_idfs(idf_file)
print len(vocab_dict), word_vecs.shape

fu = FileUtils()
onto_folder = "bioontologies/"
vec_folder = "onto_vectors_skospref/"
unmapped_folder = "unmapped/"
mfio = MatrixIO()
all_onto_files = fu.get_reqd_fileset(onto_folder, lambda x: False if ".json" in x else True)
for k in all_onto_files:
    print "starting " + k
    onto_embeddings, all_unmapped = generate_onto_vectors(onto_folder + k)
    np.save(vec_folder + k, onto_embeddings)
    mfio.save_matrix(all_unmapped, unmapped_folder + k + ".dict")
# In[211]:
#onto_file = "meddra.ttl.json"




