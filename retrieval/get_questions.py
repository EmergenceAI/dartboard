import requests, os

LLAMA_URL = 'https://api.platform.merlynmind.ai/meta-llama-llama-2-70b-chat/v1/generate'

PROMPT_STR_QUERY = """
You are given a document as follows:

[Start Document]

{document}

[End Document]

Generate a set of 10 questions and their corresponding answers given the content of the document such that the questions can be answered using only the content of the document alone.

For example, if the document is about the history of the United States the response should be:
Question 1: What is the capital of the United States?
Answer 1: Washington, D.C.
Question 2: Who was the first president of the United States?
Answer 2: George Washington
"""

import openai
from diskcache import Cache
HOME_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME_DIR, './.cache/merlyn/dartboard_llama/')
cache = Cache(os.path.join(DATA_DIR, 'cache_encoder'), eviction_policy='none') # This is a directory.
cache.reset('cull_limit', 0)
@cache.memoize()
def call_gpt(prompt: str, model: str='gpt-3.5-turbo-instruct') -> str:
    if call_gpt.client is None: call_gpt.client = openai.OpenAI()
    response = openai.completions.create(model=model, prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.)
    return response.choices[0].text.strip()
call_gpt.client = None

## Uncomment the next several lines to allow caching of the Llama2-70b API responses
def call_llama(prompt: str) -> str:
    if call_llama.api_key is None: call_llama.api_key = os.getenv('LLAMA_API_KEY')
    headers = {'Content-Type': 'application/json'}
    data = {'parameters': {
        "do_sample": False,
        "top_p": 0.99,
        "temperature": 0.01, # We should be able to do 0 since do_sample is False
        "top_k": 20,         # See https://github.com/meta-llama/llama/issues/687
        "num_return_sequences": 1,
        "max_new_tokens": 1024,
        "stop": ["<|endoftext|>", "</s>"],
        "repetition_penalty": 1
    }}
    data['prompt'] = prompt
    response = requests.post(LLAMA_URL, headers=headers, json=data, params={'key': call_llama.api_key})
    return response.json()['generated_text']
call_llama.api_key = None
