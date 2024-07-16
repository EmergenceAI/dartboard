import numpy as np, os, json, tqdm, yaml, pickle, matplotlib.pyplot as plt, random
from collections import Counter
from dartboard import encode, get_dists_cossim, get_dists_crosscoder, get_dartboard_crosscoder2, get_mmr_crosscoder2, get_knn, get_knn_crosscoder
from dartboard import get_dists_hybrid, get_dartboard_hybrid
from RGB import predict, get_scores, get_model, get_args

def processdata_broad(query, texts, embs, maxk, dorandom=True):
    # Being lazy...
    results = {}
    maxparam, step = 1, .001
    triage = 100
    for sim_getter in [get_dists_cossim, get_dists_crosscoder]:
        get_dists_results = sim_getter(query, embs, encode, texts, triage)
        for param in np.arange(0, maxparam+step, step):
            if 0 <= param <= 1:
                results[(f'mmr{sim_getter.__name__}', query, param)] = get_mmr_crosscoder2(get_dists_results, param, maxk)
            results[(f'dartboard{sim_getter.__name__}', query, param)] = get_dartboard_crosscoder2(get_dists_results, param, maxk)
    results[('knn_crosscoder', query)] = get_knn_crosscoder(query, embs, encode, texts, maxk, triage)
    results[('knn_cossim', query)] = get_knn(embs, encode(query), triage)[:maxk]
    results[('empty', query)] = []
    if dorandom:
        for seed in range(10):
            random.seed(seed)
            results[('random', query, seed/10.)] = random.sample(list(texts.keys()), maxk)
    get_dists_results_hybrid = get_dists_hybrid(query, embs, encode, texts, triage)
    for param in np.arange(0, maxparam+step, step):
        results[('dartboard_hybrid', query, param)] = get_dartboard_hybrid(get_dists_results_hybrid, param, maxk)
    # Actually put the texts in place of the indices
    results_texts = {key: [texts[str(i)] for i in indices] for key, indices in results.items()}
    return results_texts

def processdata_oracle(instance, maxk, integration):
    texts = []
    if integration:
        # Round robin append elements from instance['positive'], which is a list of lists.
        lislis2 = [lis.copy() for lis in instance['positive']]
        all_empty = False
        while not all_empty:
            all_empty = True
            for lis in lislis2:
                if lis:
                    texts.append(lis.pop(0))
                    all_empty = False
    else:
        texts = instance['positive']
    texts += instance['negative']
    return texts[:maxk]

################################################################
def dkeyunpack(dkey): return (dkey[:2] + (0.,)) if len(dkey) <= 2 else dkey

def get_first_index(lis, items):
    for i, item in enumerate(lis):
        if item in items: return i
    return np.inf
def get_single_ndcg(lis, items):
    first_index = get_first_index(lis, items)
    if first_index == np.inf: return 0
    return 1/np.log2(first_index+2)
def get_ndcg(all_results, instances, integration):
    ndcg_scores = Counter()
    for idx in all_results:
        positives = instances[idx]['positive']
        for dkey in all_results[idx]:
            method, _query, param = dkeyunpack(dkey)
            if integration:
                for positives2 in positives:
                    ndcg_scores[(method, param)] += get_single_ndcg(all_results[idx][dkey], positives2)
            else:
                ndcg_scores[(method, param)] += get_single_ndcg(all_results[idx][dkey], positives)
    return ndcg_scores

def plot_curve(data):
    curves = {}
    for score, name, param in data:
        if name not in curves: curves[name] = []
        curves[name].append((param, score))
    return curves

def get_diversity(instances, all_texts, all_embs, integration, args):
    # Get the results.
    all_embs2 = {text: all_embs[title] for title, text in all_texts.items()}
    all_results = {}
    for idx, instance in tqdm.tqdm(instances.items()):
        all_results[idx] = processdata_broad(instance['query'], all_texts, all_embs, args.passage_num)
    qembs = {}
    diversities, relevances = {}, {}
    for idx in all_results:
        for dkey in all_results[idx]:
            method, query, param = dkeyunpack(dkey)
            assert dkey not in diversities
            embs = [all_embs2[x] for x in all_results[idx][dkey]]
            if query not in qembs: qembs[query] = encode(query)
            qemb = qembs[query]
            relevances[(method, query, param)] = np.mean([np.dot(qemb, emb) for emb in embs])
            diversities[(method, query, param)] = np.mean([np.dot(emb1, emb2) for emb1 in embs for emb2 in embs])
    diversities2, relevances2 = {}, {}
    for method, query, param in diversities:
        if (method, param) not in diversities2:
            diversities2[(method, param)] = []
            relevances2[(method, param)] = []
        diversities2[(method, param)].append(diversities[(method, query, param)])
        relevances2[(method, param)].append(relevances[(method, query, param)])
    diversities3, relevances3 = {}, {}
    for method, param in diversities2:
        diversities3[method, param] = np.mean(diversities2[method, param])
        relevances3[method, param] = np.mean(relevances2[method, param])
    diversities4 = {}
    for method, param in diversities3:
        if method not in diversities4: diversities4[method] = []
        diversities4[method].append((param, 1.-diversities3[method, param], relevances3[method, param]))

    # Make two plots above each other.  First is diversity, second is relevance.
    methods = [
        ('mmrget_dists_cossim', 'mmr cosine'),
        ('dartboardget_dists_cossim', 'dartboard cosine'),
        ('mmrget_dists_crosscoder', 'mmr cross-encoder'),
        ('dartboardget_dists_crosscoder', 'dartboard cross-encoder'),
        ('dartboard_hybrid', 'dartboard hybrid'),
    ]
    hlines = [
        ('knn_crosscoder', 'knn cross-encoder'),
        ('knn_cossim', 'knn cosine'),
    ]
    plt.clf()
    plt.figure(figsize=(10, 10))
    for method, label in methods:
        data = sorted(diversities4[method], key=lambda x: x[0])
        plt.plot([x[0] for x in data], [x[1] for x in data], label=label)
    for method, label in hlines:
        assert len(diversities4[method]) == 1
        plt.axhline(diversities4[method][0][1], label=label, linestyle='dotted')
    plt.legend()
    plt.xlabel('Parameter')
    plt.ylabel('Diversity')
    plt.show()

    if False:
        plt.clf()
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        for method, label in methods:
            data = sorted(diversities4[method], key=lambda x: x[0])
            plt.plot([x[0] for x in data], [x[1] for x in data], label=label)
        for method, label in hlines:
            assert len(diversities4[method]) == 1
            plt.axhline(diversities4[method][0][1], label=label, linestyle='dotted')
        plt.legend()
        plt.xlabel('Parameter')
        plt.ylabel('Diversity')
        plt.subplot(2, 1, 2)
        for method, label in methods:
            data = sorted(diversities4[method], key=lambda x: x[0])
            plt.plot([x[0] for x in data], [x[2] for x in data], label=label)
        for method, label in hlines:
            assert len(diversities4[method]) == 1
            plt.axhline(diversities4[method][0][2], label=label, linestyle='dotted')
        plt.legend()
        plt.xlabel('Parameter')
        plt.ylabel('Relevance')
        plt.show()

def run_args(args):
    integration = 'int' in args.dataset
    # Load the model and prompt.
    prompt = yaml.load(open(f'config/instruction{"_fact" if args.factchecking else ""}.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)[args.dataset[:2]]
    system, instruction = prompt['system'], prompt['instruction']
    model = get_model(args.modelname, args)
    resultpath = 'result-en' if 'en' in args.dataset else 'result-zh'
    if not os.path.exists(resultpath): os.mkdir(resultpath)
    if args.factchecking: resultpath += '/fact'

    # Retrieve the passages.
    instances = [json.loads(line) for line in open(f'data/{args.dataset}.json', 'r', encoding='utf-8')]
    instances = {instance['id']: instance for instance in instances}

    # Use all the passages.
    all_passages = []
    for instance in instances.values():
        if integration:
            for positives in instance['positive']: all_passages += positives
        else: all_passages += instance['positive']
        all_passages += instance['negative']
    all_texts = {str(i) : all_passages[i] for i in range(len(all_passages))}
    all_embs = {title: encode(text) for title, text in all_texts.items()}

    # Get the NDCG scores.
    all_results_ndcg = {}
    ndcg_num = 40
    for idx, instance in tqdm.tqdm(instances.items()):
        all_results_ndcg[idx] = processdata_broad(instance['query'], all_texts, all_embs, ndcg_num)
        all_results_ndcg[idx][('oracle', instance['query'])] = processdata_oracle(instance, ndcg_num, integration)
    ndcg_scores = get_ndcg(all_results_ndcg, instances, integration)
    data_ndcg = [(v, method, param) for (method, param), v in ndcg_scores.items()]
    data_ndcg.sort()

    # Get the results.
    all_results = {}
    for idx, instance in tqdm.tqdm(instances.items()):
        all_results[idx] = processdata_broad(instance['query'], all_texts, all_embs, args.passage_num)
        all_results[idx][('oracle', instance['query'])] = processdata_oracle(instance, args.passage_num, integration)

    # Predict the answers.
    donedocsfname = f'{resultpath}/donedocs_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}.pkl'
    donedocs = pickle.load(open(donedocsfname, 'rb')) if os.path.exists(donedocsfname) else {}
    for idx in tqdm.tqdm(all_results):
        instance = instances[idx]
        ans = instance['answer']
        for dkey in all_results[idx]:
            query = dkey[1]
            docs = all_results[idx][dkey]
            docs2 = tuple(docs)
            if (idx, docs2) not in donedocs:
                donedocs[(idx, docs2)] = predict(query, ans, docs, model, system, instruction, args.temp, args.dataset)
    # Pickle donedocs
    with open(donedocsfname, 'wb') as f: pickle.dump(donedocs, f)

    # Total hack to get the keys!
    allkeys = set()
    for idx in all_results:
        for dkey in all_results[idx]:
            method, query, param = dkeyunpack(dkey)
            allkeys.add((method, param))
    # This is slow.  Would be more efficient to reverse the dictionary first.
    all_scores = {}
    for method, param in allkeys:
        results = []
        for idx in all_results:
            instance = instances[idx]
            ans = instance['answer']
            query = instance['query']
            dkey = (method, query, param) if param is not None else (method, query)
            if dkey not in all_results[idx]: dkey = (method, query)
            if dkey not in all_results[idx]: continue
            docs = all_results[idx][dkey]
            docs2 = tuple(docs)
            label, prediction, factlabel = donedocs[(idx, docs2)]
            newinstance = {'id': instance['id'], 'query': query, 'ans': ans, 'label': label, 'prediction': prediction,
                           'docs': docs, 'noise_rate': args.noise_rate, 'factlabel': factlabel}
            results.append(newinstance)
        # Score the answers.
        scores = get_scores(results, args)
        all_scores[(method, param)] = scores
    data_end2end = [(float(scores['all_rate']), method, (0. if param is None else param)) for (method, param), scores in all_scores.items()]
    data_end2end.sort()
    curveses = {}
    # curveses['end2end'] = plot_curve(data_end2end)
    # curveses['ndcg'] = plot_curve(data_ndcg)
    curveses['end2end'] = data_end2end
    curveses['ndcg'] = data_ndcg
    return curveses

def make_plots(curveseses):
    namemap = {
        'dartboard_hybrid2':              'Dartboard hybrid',
        'dartboardget_dists_cossim':      'Dartboard',
        'dartboardget_dists_crosscoder':  'Dartboard Crosscoder',
        'knn_cossim':                     'KNN',
        'knn_crosscoder':                 'KNN Crosscoder',
        'mmrget_dists_cossim':            'mmr',
        'mmrget_dists_crosscoder':        'mmr Crosscoder',
    }
    for dataset, datas in curveseses.items():
        for cname, curves in datas.items():
            plt.clf()
            datas1 = datas[cname]
            curves = plot_curve(datas1)
            print(f'{cname} {dataset}')
            for name, curve in sorted(curves.items()):
                print(f'{max([v for k, v in curve]):8.4f} max for {name:>30} at {max([(v,k) for k, v in curve])[1]:8.4f}')
                if len(curve) > 1 and name != 'random':
                    curve.sort()
                    x, y = zip(*curve)
                    plt.plot(x, y, label=namemap.get(name, name))
                    # plt.text(x[-1]-.1, y[-1], namemap.get(name, name), fontsize=12)
            plt.title(f'{dataset} {cname}')
            plt.legend()
            # Make the figure huge!
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            # pickle.dump(fig , open(f'plots/{dataset}_{cname}.pkl', 'wb'))
            plt.show()
            plt.savefig(f'plots/{dataset}_{cname}.png')
        # Plot curves1 against curves2
        curves1 = plot_curve(datas['ndcg'])
        curves2 = plot_curve(datas['end2end'])
        plt.clf()
        for i, (name, curve) in enumerate(curves1.items()):
            if len(curve) <= 1: continue
            if name == 'random': continue
            scatter = []
            suffix = namemap.get(name, name)
            curve2 = curves2[name]
            curve.sort()
            curve2.sort()
            x1, y1 = zip(*curve)
            x2, y2 = zip(*curve2)
            assert len(x1) == len(x2)
            for j in range(len(x1)):
                assert np.isclose(x1[j], x2[j])
                scatter.append((y1[j], y2[j], i, suffix))
            x, y, _c, _s = zip(*scatter)
            plt.scatter(x, y, label=suffix)
        # Show the legend with legend to color (range(len(legend)))
        plt.title(f'{dataset} scatter')
        plt.xlabel('ndcg score')
        plt.ylabel('end2end score')
        plt.legend(frameon=True)
        plt.show()
        plt.savefig(f'plots/{dataset}_scatter.png')

def main():
    # args.plm = 'THUDM/chatglm-6b'
    # args.api_key = os.environ['OPENAI_API_KEY']
    args = get_args()
    args.temp = 0.01 # temperature
    args.passage_num = 5
    args.modelname = 'chatglm' # chatgpt Llama-2 chatglm moss vicuna Qwen Baichuan WizardLM BELLE

    resultsfname = f'curveseses_{args.temp}_{args.passage_num}_{args.modelname}.pkl'
    if os.path.exists(resultsfname):
        curveseses = pickle.load(open(resultsfname, 'rb'))
    else:
        curveseses = {}
        for dataset in ['en', 'en_int']:
            args.dataset = dataset
            curveseses[dataset] = run_args(args)
        # Pickle the curveseses
        with open(resultsfname, 'wb') as f: pickle.dump(curveseses, f)

    make_plots(curveseses)
