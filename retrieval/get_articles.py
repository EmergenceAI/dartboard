#!/usr/bin/env python3
import requests, pandas as pd, os, re, time, subprocess, json, tiktoken, difflib, numpy as np, shutil
from get_questions import call_gpt, PROMPT_STR_QUERY

GCPBUCKET = 'zulu/ai-algos'
DATADIR = 'data/dartboard/data'
RAWDIR   = 'wikipedia_revs_raw'
CLEANDIR = 'wikipedia_revs_clean'
FINALDIR = 'wikirevs'
TOP_TITLES_FNAME = 'wikititles.csv'
NUM_VERSIONS = 5
REV_SKIP = 5
MIN_TOKENS = 50
MAX_TOKENS = 2048
#
HOMEDIR = os.path.expanduser('~')
AIBUCKET = os.path.join(HOMEDIR, GCPBUCKET)
DATAPATH = os.path.join(AIBUCKET, DATADIR)

################################################################ Downloading articles
PARAMS = {
    'action': 'query',
    'prop': 'revisions',
    'rvlimit': f'{NUM_VERSIONS*REV_SKIP}',
    'rvprop': 'timestamp|user|comment|content',
    'rvdir': 'older',
    'rvstart': '2024-01-01T00:00:00Z',
    'rvend': '2014-01-01T00:00:00Z',
    'rvslots': 'main',
    'formatversion': '2',
    'format': 'json'
}
URL = 'https://en.wikipedia.org/w/api.php'

################

def download_article(title: str, S: requests.Session):
    params = PARAMS.copy()
    params['titles'] = title
    title_path = os.path.join(DATAPATH, RAWDIR, title.replace('/', '_'))
    if os.path.exists(title_path): return
    R = S.get(url=URL, params=params)
    data = R.json()
    os.makedirs(title_path)
    revs = data['query']['pages'][0].get('revisions', [])
    for rev in range(NUM_VERSIONS):
        print(f'{title} - {rev}', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        if rev * REV_SKIP >= len(revs): break
        try: text = revs[rev * REV_SKIP]['slots']['main']['content']
        except KeyError: continue
        try:
            # Pandoc sometimes hangs when converting text, so we need to set a timeout
            output = subprocess.check_output(['pandoc', '-f', 'mediawiki', '-t', 'plain', '--wrap=none'], input=text.encode('utf-8'), timeout=10)
            text = output.decode('utf-8')
        # subprocess.CalledProcessError, subprocess.TimeoutExpired
        except Exception as e: #pylint: disable=broad-except
            print(f'Error converting text: {e}, exception type is {type(e)}')
            continue
        timestamp = revs[rev*REV_SKIP]['timestamp']
        # Save the text in a file with the number of the revision in the 'dataset' folder
        outname = f'{title.replace("/", "_")}-{timestamp}.txt'
        with open(os.path.join(title_path, outname), 'w', encoding='utf-8') as f: f.write(text)

def download_raw_wikipedia(num_articles: int, top_titles_fname: str, verbose: bool=False):
    fname = os.path.join(DATAPATH, top_titles_fname)
    top_10k_list = pd.read_csv(fname, header=None)[0].tolist()[:num_articles]
    S = requests.Session()
    for title in top_10k_list:
        if verbose: print(title)
        download_article(title, S)

################################################################ Cleaning articles
def clean_text(text: str):
    # Remove references "[1]", "[2]", etc. and "[citation needed]"
    text = re.sub(r'\[\d*\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text)
    # Remove the "See also" section and below
    text = re.sub(r'\nSee also\n.*', '', text, flags=re.DOTALL)
    return text

def token_trim(text: str, min_tokens: int, max_tokens: int):
    if token_trim.tokenizer is None: token_trim.tokenizer = tiktoken.get_encoding('cl100k_base')
    tokenized = token_trim.tokenizer.encode(text)
    if len(tokenized) < min_tokens: return ''
    if len(tokenized) > max_tokens:
        return token_trim.tokenizer.decode(tokenized[:max_tokens])
    return text
token_trim.tokenizer = None

def clean_articles(min_tokens: int=MIN_TOKENS, max_tokens: int=MAX_TOKENS, verbose: bool=False):
    for dname in os.listdir(os.path.join(DATAPATH, RAWDIR)):
        if verbose: print(dname)
        if not os.path.isdir(os.path.join(DATAPATH, RAWDIR, dname)): continue
        # Make sure there are at least NUM_VERSIONS files in the directory
        dirs = os.listdir(os.path.join(DATAPATH, RAWDIR, dname))
        if len(dirs) < NUM_VERSIONS:
            if verbose: print(f'Cleaning: Not enough files in {dname}')
            continue
        # Clean each file and write it to the same directory in CLEANDIR
        os.makedirs(os.path.join(DATAPATH, CLEANDIR, dname), exist_ok=True)
        for fname in dirs:
            with open(os.path.join(DATAPATH, RAWDIR, dname, fname), 'r', encoding='utf-8') as f: text = f.read()
            text = clean_text(text)
            if min_tokens > 0 and max_tokens > 0:
                text = token_trim(text, min_tokens, max_tokens)
            # Write the cleaned text to a new file
            if text:
                with open(os.path.join(DATAPATH, CLEANDIR, dname, fname), 'w', encoding='utf-8') as f: f.write(text)

################################################################ Delta Delta Delta, can I help ya help ya help ya?
def get_delta(text1: str, text2: str, verbatim: bool=False):
    delta = 0
    for i, s in enumerate(difflib.ndiff(text1, text2)):
        if s[0]==' ': continue
        if verbatim:
            if s[0]=='-': print(u'Delete "{}" from position {}'.format(s[-1],i))
            elif s[0]=='+': print(u'Add "{}" to position {}'.format(s[-1],i))
        delta += 1
    return delta

def get_deltas(title):
    subdir = os.path.join(DATAPATH, CLEANDIR, title.replace("/", "_"))
    dirs = sorted(os.listdir(subdir))
    # Do the full matrix of deltas
    deltas = np.zeros((len(dirs), len(dirs)))
    lengths = np.zeros(len(dirs))
    for i in range(len(dirs)):
        with open(os.path.join(subdir, dirs[i]), 'r', encoding='utf-8') as f: text1 = f.read()
        lengths[i] = len(text1)
        for j in range(i+1, len(dirs)):
            with open(os.path.join(subdir, dirs[j]), 'r', encoding='utf-8') as f: text2 = f.read()
            deltas[i, j] = get_delta(text1, text2)
            deltas[j, i] = deltas[i, j]
    # Pretty print the deltas
    return deltas, lengths

################################################################ Generating questions
def generate_queries(text):
    answers_for_queries = {}
    try:
        response = call_gpt(PROMPT_STR_QUERY.format(document=text))
    except Exception as e: #pylint: disable=broad-except
        print(f'Error in generate_queries: {e}')
        return answers_for_queries
    # Remove duplicate \n's
    response = re.sub(r'\n+', '\n', response)
    if response.startswith('Question '):
        response = response.lstrip('Question ')
        query_list = response.split('\nQuestion ')
    else:
        query_list = response.split('\n')
        # Now rejoin every other line
        query_list = [f'{query_list[i]}\n{query_list[i+1]}' for i in range(0, len(query_list), 2)]
    for q in query_list:
        q = q.split('\n')
        if len(q) == 2:
            q, a = q[0].strip(), q[1].strip()
            # Strip off 'Answer' or 'Answer:' from a
            a = re.sub(r'^Answer:?\s*', '', a)
            # Strip and remove the question number using regex 'N+: '
            q = re.sub(r'^\d+: ', '', q)
            q = re.sub(r'^\d+. ', '', q)
            a = re.sub(r'^\d+: ', '', a)
            a = re.sub(r'^\d+. ', '', a)
            answers_for_queries[q] = a
    return answers_for_queries

def get_questions(top_titles_fname: str, max_articles: int=None):
    fname = os.path.join(DATAPATH, top_titles_fname)
    top_10k_list = pd.read_csv(fname, header=None)[0].tolist()[:max_articles]
    # for dname in sorted(os.listdir(os.path.join(DATAPATH, CLEANDIR))):
    for title in top_10k_list:
        dname = title.replace("/", "_")
        if not os.path.isdir(os.path.join(DATAPATH, CLEANDIR, dname)):
            print(f'No directory for {dname}')
            continue
        print(f'Generating questions for {dname}')
        dirs = sorted(os.listdir(os.path.join(DATAPATH, CLEANDIR, dname)))
        if len(dirs) < NUM_VERSIONS:
            print(f'Not enough files in {dname}')
            continue
        fname = dirs[-1] # use only the latest version
        with open(os.path.join(DATAPATH, CLEANDIR, dname, fname), 'r', encoding='utf-8') as f: text = f.read()
        # Generate the questions
        try: queries = generate_queries(text)
        except Exception as e: #pylint: disable=broad-except
            print(f'Error in get_questions: {e}')
            continue
        if len(queries) < 10: print(f'Only {len(queries)} questions generated for {dname}')
        # Queries is a dictionary with the question as the key and the answer as the value
        # Write the questions to a file
        with open(os.path.join(DATAPATH, CLEANDIR, f'{dname}.json'), 'w', encoding='utf-8') as f: json.dump(queries, f, indent=2)

################################################################ Final
def make_single_dir(top_titles_fname: str, max_articles: int=10000):
    fname = os.path.join(DATAPATH, top_titles_fname)
    top_10k_list = pd.read_csv(fname, header=None)[0].tolist()
    num_articles = 0
    i = 0
    all_queries = {}
    # Start with a clean directory os.path.join(DATAPATH, FINALDIR)
    shutil.rmtree(os.path.join(DATAPATH, FINALDIR), ignore_errors=True)
    os.makedirs(os.path.join(DATAPATH, FINALDIR), exist_ok=True)
    while num_articles < max_articles:
        title = top_10k_list[i]
        i += 1
        dname = title.replace("/", "_")
        subdir = os.path.join(DATAPATH, CLEANDIR, dname)
        if not os.path.isdir(subdir):
            print(f'No directory for {title}')
            continue
        dirs = sorted(os.listdir(os.path.join(DATAPATH, CLEANDIR, dname)))
        if len(dirs) < NUM_VERSIONS:
            print(f'Not enough files in {dname}')
            continue
        jsonf = os.path.join(DATAPATH, CLEANDIR, f'{dname}.json')
        if not os.path.isfile(jsonf):
            print(f'No json file for {dname}')
            continue
        # Get the json
        with open(jsonf, 'r', encoding='utf-8') as f: queries = json.load(f)
        if len(queries) < 5:
            print(f'Only {len(queries)} questions generated for {dname}, skipping')
            continue
        all_queries[dname] = queries
        # Copy the directory to the new location
        num_articles += 1
        # print(f'Copying {subdir} to {dname}')
        if len(dirs) > NUM_VERSIONS:
            print(f'Warning: {dname} has {len(dirs)} versions')
        for fname in dirs:
            # Check if the file is already there.
            if os.path.isfile(os.path.join(DATAPATH, FINALDIR, dname, fname)):
                print(f'Warning: File {fname} already exists in {dname}')
            shutil.copy(os.path.join(subdir, fname), os.path.join(DATAPATH, FINALDIR))
    # Remove duplicate queries
    dupes = {}
    for dname in all_queries:
        for q in all_queries[dname]:
            if q not in dupes: dupes[q] = []
            dupes[q].append(dname)
    for q in dupes:
        if len(dupes[q]) > 1:
            for dname in dupes[q]: all_queries[dname].pop(q)
    all_queries2 = {}
    for dname in all_queries:
        for q in all_queries[dname]:
            assert q not in all_queries2
            all_queries2[q] = (all_queries[dname][q], dname)
    # Write the jsonl file
    with open(os.path.join(DATAPATH, 'all_queries.json'), 'w', encoding='utf-8') as f: json.dump(all_queries2, f, indent=2)

################################################################ Embeddings
import sentence_transformers as st, pickle
from diskcache import Cache
EMBS_DATA_DIR = os.path.join(HOMEDIR, './.cache/merlyn/dartboard_embs/')
cache2 = Cache(os.path.join(EMBS_DATA_DIR, 'cache_encoder'), eviction_policy='none') # This is a directory.
cache2.reset('cull_limit', 0)

@cache2.memoize()
def encode(dstr: str):
    if encode.encoder is None: encode.encoder = st.SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    return encode.encoder.encode(dstr)
encode.encoder = None

def encode_all_articles():
    # Encode all the articles in the final directory
    embs = {}
    embs_file = os.path.join(DATAPATH, 'embs.pkl')
    # Load pickle file if it exists
    if os.path.isfile(embs_file):
        with open(embs_file, 'rb') as f: embs = pickle.load(f)
    final_dir = os.path.join(DATAPATH, FINALDIR)
    dirs = sorted(os.listdir(final_dir))
    for fname in dirs:
        if fname in embs: continue
        print(f'Encoding {fname}')
        with open(os.path.join(final_dir, fname), 'r', encoding='utf-8') as f: text = f.read()
        embs[fname] = encode(text)
    # Pickle the embeddings
    with open(embs_file, 'wb') as f: pickle.dump(embs, f)
    return embs

def get_all_article_texts():
    texts = {}
    final_dir = os.path.join(DATAPATH, FINALDIR)
    for fname in sorted(os.listdir(final_dir)):
        with open(os.path.join(final_dir, fname), 'r', encoding='utf-8') as f: text = f.read()
        texts[fname] = text
    return texts

def split_base_version(dname: str) -> str:
    # split into base and version
    base = re.sub(r'-\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\.txt$', '', dname)
    version = re.sub(r'^.*?-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\.txt$', r'\1', dname)
    return base, version

from hashlib import sha256
def get_all_articles_dedup():
    # Encode all the articles in the final directory
    names = {}
    final_dir = os.path.join(DATAPATH, FINALDIR)
    dirs = sorted(os.listdir(final_dir))
    for fname in dirs:
        with open(os.path.join(final_dir, fname), 'r', encoding='utf-8') as f: text = f.read()
        sha = sha256(text.encode()).hexdigest()
        if sha in names:
            if split_base_version(names[sha])[0] != split_base_version(fname)[0]:
                print(f'Warning: Duplicate hash for {fname} and {names[sha]}')
        names[sha] = fname
    return set(names.values())

def dedup_articles(texts):
    names = {}
    for fname, text in texts.items():
        sha = sha256(text.encode()).hexdigest()
        if sha in names:
            if split_base_version(names[sha])[0] != split_base_version(fname)[0]:
                print(f'Warning: Duplicate hash for {fname} and {names[sha]}')
        names[sha] = fname
    return set(names.values())

################################################################ HF stuff
from datasets import Dataset, DatasetDict, load_dataset # noqa: E402
HFDATASETNAME = 'MerlynMind/atg-wikipedia-qa'

def load_without_hf():
    embs = encode_all_articles()
    texts = get_all_article_texts()
    with open(os.path.join(DATAPATH, 'all_queries.json'), 'r', encoding='utf-8') as f: all_queries = json.load(f)
    return all_queries, embs, texts

def dump_to_hf(all_queries, embs, texts):
    hftoken = os.getenv('HUGGINGFACE_TOKEN')
    queries2 = [{'title': q,     'text': '',   'emb': np.array(0, dtype=np.float32), 'query': all_queries[q]} for q in all_queries]
    embs2    = [{'title': title, 'text': '',   'emb': emb,                           'query': ('', '')      } for title, emb in embs.items()]
    texts2   = [{'title': title, 'text': text, 'emb': np.array(0, dtype=np.float32), 'query': ('', '')      } for title, text in texts.items()]
    DatasetDict({
        'queries': Dataset.from_list(queries2),
        'embs': Dataset.from_list(embs2),
        'texts': Dataset.from_list(texts2),
    }).push_to_hub(HFDATASETNAME, token=hftoken) # Updload

def load_from_hf():
    hftoken = os.getenv('HUGGINGFACE_TOKEN')
    queries1 = load_dataset(HFDATASETNAME, split='queries', token=hftoken)
    all_queries = {d['title']: d['query'] for d in queries1}
    embs1 = load_dataset(HFDATASETNAME, split='embs', token=hftoken)
    embs = {d['title']: d['emb'] for d in embs1}
    texts1 = load_dataset(HFDATASETNAME, split='texts', token=hftoken)
    texts = {d['title']: d['text'] for d in texts1}
    return all_queries, embs, texts

################################################################ The Main Event
def main():
    download_raw_wikipedia(num_articles=20000, top_titles_fname=TOP_TITLES_FNAME)
    clean_articles(MIN_TOKENS, MAX_TOKENS)
    get_questions(TOP_TITLES_FNAME, max_articles=20000)
    make_single_dir(TOP_TITLES_FNAME, max_articles=10000)
    # TODO verify that *any* version of the article should be able to answer the question
    # TODO check that question is not answered by multiple articles
    _embs = encode_all_articles()

# Copy with
# cd ~/ai-algos/data/dartboard/
# gsutil -m cp -r data  gs://ai-algos/data/dartboard/

if __name__ == '__main__': main()

