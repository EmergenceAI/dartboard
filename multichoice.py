from dartboard import encode, get_knn_crosscoder, get_dartboard_crosscoder2, get_dists_crosscoder

def main():
    query = 'Do you want to watch soccer?'
    texts = [
        'Absolutely!',
        'Affirmative!',
        'I don\'t know!',
        'I\'d love to!',
        'Maybe later.',
        'Maybe!',
        'Maybe...',
        'No thanks.',
        'No way!',
        'No, I don\'t wanna do dat.',
        'No, thank you!',
        'No, thank you.',
        'Not right now.',
        'Not today.',
        'Perhaps..',
        'Sure!',
        'Yeah!',
        'Yes!',
        'Yes, please can we?',
        'Yes, please!',
        'Yes, please.',
        'Yes, we ought to!',
        'Yes, we shall!',
        'Yes, we should!',
    ]

    texts = {i+1: text for i, text in enumerate(texts)}
    embs = {i: encode(text) for i, text in texts.items()}
    triage = 100
    sigma = .5
    k = 3

    print()
    print('Candidates:')
    for title, text in texts.items(): print(f'  {title:2}: {text}')

    print()
    print('Query:', query)

    results = get_knn_crosscoder(query, embs, encode, texts, k, triage)
    print()
    print('KNN crosscoder:')
    for title in results: print(f'  {title:2}: {texts[title]}')

    dists = get_dists_crosscoder(query, embs, encode, texts, triage)
    results = get_dartboard_crosscoder2(dists, sigma, k)
    print()
    print('Dartboard crosscoder:')
    for title in results: print(f'  {title:2}: {texts[title]}')
