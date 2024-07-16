import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from mmr import my_mmr

def norm(dist, sigma):
    if sigma < 1e-9: return 0. * dist # return zero in shape of dist if dist is np array
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-dist**2 / (2 * sigma**2))

def score_darts_np(query, guesses, vectors, sigma=1.):
    guesses = np.array(guesses)
    qv = norm(np.linalg.norm(query-vectors, axis=1), sigma)
    # diff between query and vectors, where g[i][j] = query[i] - vectors[j]
    g = guesses[:, None] - vectors
    norm_g = np.linalg.norm(g, axis=2)
    max_g = np.max(norm(norm_g, sigma), axis=0)
    return np.sum(qv * max_g)

def dartboard(query, vectors, top_n, sigma, sim_metric=euclidean_distances):
    vectors = np.array(vectors)
    maxi = np.argmax(-sim_metric([query], vectors)[0])
    guesses = [vectors[maxi]]
    idxs = [maxi]
    for _ in range(top_n-1):
        best_guess, best_score = None, None
        for i, v in enumerate(vectors):
            score = score_darts_np(query, guesses + [v], vectors, sigma)
            if best_score is None or score > best_score: best_guess, best_score = i, score
        guesses.append(vectors[best_guess])
        idxs.append(best_guess)
    return vectors[idxs]

# For the main illustrative plot
def main():
    top_n = 5
    num_points = 400
    np.random.seed(42)
    vectors = list(np.random.rand(num_points, 2))
    query = np.array([.41, .67])
    sigma = .1
    guesses = dartboard(query, vectors, top_n, sigma)
    knn = my_mmr(query, vectors, top_n, 0., sim_metric=lambda x, y=None: -euclidean_distances(x, y))
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    for rad in np.arange(sigma, sigma*10, sigma):
        ax.add_artist(plt.Circle(query, rad, color=(1, 0, 0, sigma / rad), fill=False, lw=2))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.scatter([v[0] for v in knn], [v[1] for v in knn], c='#aaaaaa', marker='o', s=100)
    ax.scatter([v[0] for v in guesses], [v[1] for v in guesses], marker='o', s=200, facecolors='none', lw=2, edgecolors='g')
    for i, v in enumerate(guesses):
        ax.annotate(str(i+1), (v[0]-.01, v[1]+.01), fontsize=18, color='g', ha='center', va='center')
    ax.scatter([v[0] for v in vectors], [v[1] for v in vectors], c='b', marker='o', s=10)
    ax.scatter(query[0], query[1], c='r', marker='*', s=200)
    ax.set_aspect('equal', adjustable='box')
    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()
    # plt.xlim(.2, .6); plt.ylim(.4, .8)
    plt.show()
