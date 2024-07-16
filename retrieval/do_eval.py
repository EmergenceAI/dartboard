import random, pickle, os, numpy as np
from collections import Counter
from hashlib import sha256
from tqdm import tqdm
from get_articles import load_from_hf, dedup_articles, split_base_version
from dartboard import encode
from dartboard import get_dists_cossim, get_dists_crosscoder, get_dartboard_crosscoder2, get_mmr_crosscoder2
from dartboard import get_dists_crosscoder_hybrid, get_dartboard_crosscoder_hybrid
from matplotlib import pyplot as plt

QHASHVAL = '57cd677ed231e6576f8ab9008c6bbeb49ee065abbe1307468f0b39c3a8c17ff9'
QHASHVAL2 = 'd91a536061b568ff8d58137eea1ddb78bea0b4daf9f827af6d754f208076e52b'
def get_queries(all_queries):
    num_test = 1000
    num_eval = 100
    queries = list(all_queries.keys())
    random.seed(42)
    random.shuffle(queries)
    eval_queries = queries[num_test:num_test+num_eval]
    eval_queries.sort()
    queries = queries[:num_test]
    queries.sort()
    hashval = sha256('\n'.join(queries).encode('utf-8')).hexdigest()
    assert hashval == QHASHVAL, 'Queries subsample differs from original.'
    hashval2 = sha256('\n'.join(eval_queries).encode('utf-8')).hexdigest()
    assert hashval2 == QHASHVAL2, 'Eval queries subsample differs from original.'
    assert len(set(queries) & set(eval_queries)) == 0, 'Queries and eval_queries overlap.'
    return {q: all_queries[q] for q in queries}, {q: all_queries[q] for q in eval_queries}

def parameter_sweep(queries, embs, texts, maxk, triage, sim_getter):
    # sim_getter is get_dists_cossim or get_dists_crosscoder
    step = .01
    results = {}
    for query in tqdm(queries):
        get_dists_results = sim_getter(query, embs, encode, texts, triage)
        for sigma in np.arange(0, 1+step, step):
            results[('dartboard', query, sigma)] = get_dartboard_crosscoder2(get_dists_results, sigma, maxk)
            results[('dartboardhybrid', query, sigma)] = get_dartboard_crosscoder_hybrid(get_dists_results, sigma, maxk)
        for diversity in np.arange(0, 1+step, step):
            results[('mmr', query, diversity)] = get_mmr_crosscoder2(get_dists_results, diversity, maxk)
    return results

def get_rank(lis, item): return lis.index(item) if item in lis else np.inf

def run_queries(results, embs, embs_nodupe, queries, texts, maxk, triage):
    # crosscoder or cossim X dupe or dedupe
    all_results = {}
    for sim_getter in [get_dists_crosscoder_hybrid, get_dists_cossim, get_dists_crosscoder]:
        for dedup in [False, True]:
            fname = f'{results}/{sim_getter.__name__}_{"dedupe" if dedup else "dupe"}.pkl'
            if os.path.exists(fname): all_results[fname] = pickle.load(open(fname, 'rb'))
            else:
                print(f'Running {sim_getter.__name__} with {"dedupe" if dedup else "dupe"}')
                embs2 = (embs_nodupe if dedup else embs)
                all_results[fname] = parameter_sweep(queries, embs2, texts, maxk, triage, sim_getter)
                pickle.dump(all_results[fname], open(fname, 'wb'))
    return all_results

def get_mrr_ndcg(results, queries):
    # MRR score (Mean Reciprocal Rank) and NDCG (normalized discounted cumulutive gain)
    mrr_scores = Counter()
    ndcg_scores = Counter()
    mrr_scores2 = Counter()
    ndcg_scores2 = Counter()
    for dkey, retrieved in results.items():
        method, query, param = dkey
        _answer, base_name = queries[query]
        bases = [split_base_version(d)[0] for d in retrieved]
        mrr_scores[(method, param)] += 1 / (1 + get_rank(bases, base_name))
        mrr_scores2[(method, param)] += (1 / (1 + get_rank(bases, base_name)))**2
        ndcg_scores[(method, param)] += 1 / np.log2(2 + get_rank(bases, base_name))
        ndcg_scores2[(method, param)] += (1 / np.log2(2 + get_rank(bases, base_name)))**2
    # Convert to lists of pairs
    mrr_plots = {}
    for name, scores2 in [('MRR', mrr_scores), ('NDCG', ndcg_scores)]:
        for dkey in sorted(scores2):
            method, param = dkey
            if (name, method) not in mrr_plots: mrr_plots[(name, method)] = []
            mrr_plots[(name, method)].append((param, scores2[dkey]))
    # The 2nd moments.
    mrr_plots2 = {}
    for name, scores2 in [('MRR', mrr_scores2), ('NDCG', ndcg_scores2)]:
        for dkey in sorted(scores2):
            method, param = dkey
            if (name, method) not in mrr_plots2: mrr_plots2[(name, method)] = []
            mrr_plots2[(name, method)].append((param, scores2[dkey]))
    return mrr_plots, mrr_plots2

def make_plots(mrr_plots, fname, show=False):
    names = sorted(set([name for name, _ in mrr_plots]))
    methods = sorted(set([method for _, method in mrr_plots]))
    for name in names:
        # Use a different x axis scale for each method, going from min to max
        for method in methods:
            params, scores, _std_errors = zip(*mrr_plots[(name, method)])
            minx, maxx = min(params), max(params)
            # Rescale to be between 0 and 1
            params = [(p - minx) / (maxx - minx) for p in params]
            plt.plot(params, scores, label=method)
            # plt.errorbar(params, scores, yerr=std_errors, fmt='o')
            # Draw a line at the best value
            best = max(scores)
            plt.axhline(best, color='black', linestyle='--')
        plt.legend()
        plt.title(f'{name} scores for different methods')
        if show: plt.show()
        else:
            plt.savefig(f'{fname}-{name}.png')
            plt.close()

def get_best_params(mrr_plots):
    return {dkey: max([(y, x) for x, y in mrr_plots[dkey]])[1] for dkey in mrr_plots}

def get_std_error(scores, scores2, n):
    stddevs = {}
    for fname in scores:
        stddevs[fname] = {}
        for dkey in scores[fname]:
            stddevs[fname][dkey] = []
            for (x1, v1), (x2, v2) in zip(scores[fname][dkey], scores2[fname][dkey]):
                assert np.isclose(x1, x2)
                stddevs[fname][dkey].append((x1, v1/n, np.sqrt((v2/n) - (v1/n)**2)/np.sqrt(n)))
    return stddevs

def main():
    all_queries, embs, texts = load_from_hf()
    queries, eval_queries = get_queries(all_queries)
    # Try also with exact duplicate removal
    names = dedup_articles(texts)
    embs_nodupe = {n: embs[n] for n in names}
    # qembs = {q: encode(q) for q in queries}
    maxk = 40
    triage = 100
    ################################################################
    all_results = run_queries('results', embs, embs_nodupe, queries, texts, maxk, triage)
    all_eval_results = run_queries('results_eval', embs, embs_nodupe, eval_queries, texts, maxk, triage)

    ################################################################ Get best sigma from eval results
    best_params_eval, scores_eval, scores_eval2 = {}, {}, {}
    for fname, results in all_eval_results.items():
        scores_eval[fname], scores_eval2[fname] = get_mrr_ndcg(results, eval_queries)
        best_params_eval[fname] = get_best_params(scores_eval[fname])
    best_params, scores, scores2 = {}, {}, {}
    for fname, results in all_results.items():
        scores[fname], scores2[fname] = get_mrr_ndcg(results, queries)
        best_params[fname] = get_best_params(scores[fname])

    std_errors = get_std_error(scores, scores2, len(queries))

    # Show predicted vs actual best parameters
    print()
    for fname in best_params:
        fname_eval = fname.replace('results', 'results_eval')
        method, dedupe = fname.split('/')[-1].split('_')[-2:]
        dedupe = dedupe.split('.')[0]
        print(fname)
        for dkey in best_params[fname]:
            print(f'predicted {best_params_eval[fname_eval][dkey]:.2f} vs actual {best_params[fname][dkey]:.2f} for {method} {dedupe}')
    print()
    for fname in best_params:
        fname_eval = fname.replace('results', 'results_eval')
        method, dedupe = fname.split('/')[-1].split('_')[-2:]
        dedupe = dedupe.split('.')[0]
        for dkey in best_params_eval[fname_eval]:
            eval_name, algoname = dkey
            xval_eval = best_params_eval[fname_eval][dkey]
            xval = best_params[fname][dkey]
            # Find the pair in scores[fname][dkey] that is x closest to xval
            # pairs = scores[fname][dkey]
            pairs = std_errors[fname][dkey]
            xvals, scores2, std_error = zip(*pairs)
            ibest = np.argmin(np.abs(np.array(xvals) - xval_eval))
            iactual = np.argmin(np.abs(np.array(xvals) - xval))
            best_guess = scores2[ibest]
            best_actual = scores2[iactual]
            print(f'predicted {best_guess:5.4f} +/- {std_error[ibest]:5.4f} vs actual {best_actual:5.4f} +/- {std_error[iactual]:5.4f} for {eval_name:4} {algoname:9} for {method:10} with {dedupe:6}')
        print()

    ################################################################ Get MRR score and NDCG
    for fname in scores: make_plots(std_errors[fname], fname, True)

    # Sample to get the average cos-similarity between documents in embs
    bigvec = np.array(list(embs.values()))
    # This is 50k x 768
    # Randomly sample it to get a 1000 x 768 matrix
    sample = bigvec[np.random.choice(bigvec.shape[0], 1000, replace=False)]
    # Now get the average cos-similarity between all pairs of documents
    from sklearn.metrics.pairwise import cosine_similarity
    sims = np.mean(cosine_similarity(sample))
    avg_dist = 1 - sims
    print(f'Average distance between documents: {avg_dist:.4f}')
    # Bigger dists means bigger sigma.

    # Sigma can also be "finetuned" from a training set of questions with known correct files.
    # Same for diversity.

    #TODO: Show relevance and diversity
    #      Especially for Dartboard.  Show that Dartboard is decent enough.
    #      Diversity = avg cos sim between all pairs of retrieved documents.
    #      Relevance = avg cos sim between query and retrieved documents.

    # Show how different classes of questions depend on diversity. ('compare X & Y')
    # report diversity:
    #   do we have 'copies' for the same document?
    #   cos-similarity between retrieved documents
    #     calibrate using distance btw two versions of the same document are
    # Relevance:
    #   Distance to query?

    # Snippet-level eval: How to map snippets?
