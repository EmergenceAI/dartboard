import numpy as np, itertools, sentence_transformers as st
from diskcache import Cache
from hashlib import sha256
from scipy.special import logsumexp
from sklearn.metrics.pairwise import cosine_similarity

################################################################ CrossEncoder with caching
cache2 = Cache('./cache/cache_embs/cache_encoder', eviction_policy='none') # This is a directory.
cache2.reset('cull_limit', 0)
@cache2.memoize()
def encode(dstr: str):
    if encode.encoder is None: encode.encoder = st.SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    return encode.encoder.encode(dstr)
encode.encoder = None
crosscoder_cache = Cache('./cache/cache_crosscoder', eviction_policy='none') # This is a directory.
crosscoder_cache.reset('cull_limit', 0)
@crosscoder_cache.memoize()
def cc_cache_hack(a, b): return get_crosscoder_dists.dists[(a, b)]
# This updates dists in place.  Can use @cache, but this works on batches.
def get_crosscoder_dists(pairs):
    if get_crosscoder_dists.dists is None:
        get_crosscoder_dists.cross_encoder = st.CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda:0')
        dists = {}
        # Fill dists with the pairs we already have in the cache.
        for dkey in crosscoder_cache:
            fname, a, b, _dunno = dkey
            if fname == f'{__name__}.cc_cache_hack':
                dists[(a, b)] = cc_cache_hack(a, b)
        # Don't update the dists until we're done loading.
        get_crosscoder_dists.dists = dists
    dists, cross_encoder = get_crosscoder_dists.dists, get_crosscoder_dists.cross_encoder
    # Can also just hash each unique item once if this is too slow.
    # This probably isn't the bottleneck right now though.
    encoded, decoded = {}, {}
    for pair in pairs:
        for a in pair:
            if a not in encoded:
                encoded[a] = sha256(a.encode()).hexdigest()
                decoded[encoded[a]] = a
    pairs = [(encoded[a], encoded[b]) for a, b in pairs]
    newpairs = list(set(pairs) - set(dists.keys()))
    if len(newpairs) > 0:
        newpairs2 = [(decoded[a], decoded[b]) for a, b, in newpairs]
        newdists = cross_encoder.predict(newpairs2)
        dists.update(zip(newpairs, newdists))
        for a, b in newpairs: cc_cache_hack(a, b) # Hack to cache the results.
    return [dists[x] for x in pairs]
get_crosscoder_dists.dists = None
get_crosscoder_dists.cross_encoder = None

################################################################
def cos_sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def get_knn(embs: dict[str, np.ndarray], qemb: np.ndarray, k=5):
    sims = {dname: cos_sim(qemb, emb) for dname, emb in embs.items()}
    return sorted(sims, key=lambda x: sims[x], reverse=True)[:k]
def lognorm(dist, sigma):
    if sigma < 1e-9: return -np.inf * dist
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - dist**2 / (2 * sigma**2)
def scaled_cosdist(a, b):
    cossim = cosine_similarity(a, b)
    return np.clip(1 - (cossim + 1)/2, 0, 1)

################################################################ Dart Using CrossEncoder: Dart it up, Brah!
def crosscoder_rescale(a):
    return (crosscoder_rescale.distmax - a)/(crosscoder_rescale.distmax - crosscoder_rescale.distmin)
crosscoder_rescale.distmin = -11.6
crosscoder_rescale.distmax = 11.4

def get_dists_crosscoder(query, embs, myencode, texts, triage):
    # Do KNN to get top triage files
    top_article_titles = get_knn(embs, myencode(query), triage)
    candidates = [texts[title] for title in top_article_titles]
    cross_scores = get_crosscoder_dists([(query, text) for text in candidates])
    pairs = list(itertools.product(candidates, repeat=2))
    mydists = get_crosscoder_dists(pairs)
    mydists = np.array(mydists).reshape((len(candidates), len(candidates)))
    mydists += mydists.T
    distsmat = crosscoder_rescale(mydists/2.)
    qdists = crosscoder_rescale(np.array(cross_scores))
    return qdists, distsmat, top_article_titles

def get_dists_cossim(query, embs, myencode, _texts, triage):
    top_article_titles = get_knn(embs, myencode(query), triage)
    embs2 = np.array([embs[title] for title in top_article_titles])
    qdists = scaled_cosdist([myencode(query)], embs2)
    distsmat = scaled_cosdist(embs2, embs2)
    return qdists[0], distsmat, top_article_titles

def greedy_dartsearch(qprobs, ccprobmat, top_article_titles, k):
    top_idx = np.argmax(qprobs)
    dset = np.array([top_idx])
    maxes = ccprobmat[top_idx]
    while len(dset) < k:
        newmaxes = np.maximum(maxes, ccprobmat)
        logscores = newmaxes + qprobs
        scores = logsumexp(logscores, axis=1)
        scores[dset] = -np.inf
        best_idx = np.argmax(scores)
        maxes = newmaxes[best_idx]
        dset = np.append(dset, best_idx)
    return [top_article_titles[i] for i in dset]

def get_dartboard_crosscoder2(get_dists_results, sigma: float, k: int):
    qdists, distsmat, top_article_titles = get_dists_results
    # Hack so that we don't have to worry about sigma = 0.
    if sigma < 1e-5: sigma = 1e-5
    qprobs = lognorm(qdists, sigma)
    ccprobmat = lognorm(distsmat, sigma)
    return greedy_dartsearch(qprobs, ccprobmat, top_article_titles, k)

################ Hybrid.  These are very similar to above, so we should probably refactor.
def get_dists_hybrid(query, embs, myencode, texts, triage):
    top_article_titles = get_knn(embs, myencode(query), triage)
    embs2 = np.array([embs[title] for title in top_article_titles])
    distsmat = scaled_cosdist(embs2, embs2)
    candidates = [texts[title] for title in top_article_titles]
    cross_scores = get_crosscoder_dists([(query, text) for text in candidates])
    qdists = np.array(cross_scores)
    return qdists, distsmat, top_article_titles

def get_dartboard_hybrid(get_dists_results_hybrid, sigma: float, k: int):
    qdists, distsmat, top_article_titles = get_dists_results_hybrid
    # Hack so that we don't have to worry about sigma = 0.
    if sigma < 1e-5: sigma = 1e-5
    qprobs = qdists/sigma - logsumexp(qdists/sigma)
    ccprobmat = np.log(1 - distsmat)
    return greedy_dartsearch(qprobs, ccprobmat, top_article_titles, k)

################################################################ MMR using CrossEncoder: MMR it up, Brah!
def get_mmr_crosscoder2(get_dists_results, diversity: float, k: int):
    qdists, distsmat, top_article_titles = get_dists_results
    # Get the top sentence, then greedily add to the set of sentences
    top_idx = np.argmin(qdists)
    dset = np.array([top_idx])
    distssofar = qdists[top_idx]
    pairwisedistsofar = 0.
    while len(dset) < k:
        newdiststotal = distssofar + qdists
        newpairwisedistsofar = pairwisedistsofar + np.sum(distsmat[dset, :], axis=0)
        a = newdiststotal/(len(dset)+1)
        b = newpairwisedistsofar/((len(dset)+1)**2)
        scores = (1-diversity) * a - diversity * b
        # Hack set the scores for the existing dset to infinity, so we don't pick them again.
        scores[dset] = np.inf
        best_idx = np.argmin(scores)
        distssofar, pairwisedistsofar = newdiststotal[best_idx], newpairwisedistsofar[best_idx]
        dset = np.append(dset, best_idx)
    return [top_article_titles[i] for i in dset]

################################################################ KNN using CrossEncoder: KNN it up, Brah!
def get_knn_crosscoder(query: str, embs: dict[str, np.array], myencode, texts: dict[str, str], k: int, triage: int):
    # Do KNN to get top triage files
    top_article_titles = get_knn(embs, myencode(query), triage)
    candidates = [texts[title] for title in top_article_titles]
    cross_inp = [(query, text) for text in candidates]
    cross_scores = get_crosscoder_dists(cross_inp)
    # Sort by cross-encoder score
    reranked = [x for _, x in sorted(zip(cross_scores, top_article_titles), reverse=True)]
    return reranked[:k]
