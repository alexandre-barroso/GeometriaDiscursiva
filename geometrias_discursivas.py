import os
import re
import gzip
import shutil
import urllib.request
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

FASTTEXT_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz'
FASTTEXT_GZ  = 'cc.pt.300.vec.gz'
FASTTEXT_VEC = 'cc.pt.300.vec'

def _reporthook(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / total_size))
    print(f'\r  {pct}%', end='', flush=True)

def ensure_fasttext():
    if os.path.exists(FASTTEXT_VEC):
        return
    if not os.path.exists(FASTTEXT_GZ):
        print(f'Baixando modelo fastText ({FASTTEXT_URL})...')
        urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_GZ, reporthook=_reporthook)
        print()
    print('Descomprimindo modelo fastText...')
    with gzip.open(FASTTEXT_GZ, 'rb') as f_in, open(FASTTEXT_VEC, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print('Modelo pronto.')

try:
    plt.style.use(['science', 'ieee'])
except OSError:
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 150,
    })

STOP_WORDS = set(stopwords.words('portuguese'))
IMPORTANT_WORDS = {'não', 'sim', 'muito', 'pouco', 'mais', 'menos'}
CHAR_PATTERN = re.compile(r'[a-záéíóúâêîôûãõçàèìòùäëïöüñ]+')
SINGLE_KEEP = {'é', 'à', 'e', 'o', 'a'}

WORD_PAIRS = [
    ('democracia', 'ditadura'),
    ('dinheiro', 'elite'),
    ('governo', 'população'),
    ('pobres', 'elite'),
    ('população', 'desemprego'),
]


def preprocess(text):
    words = word_tokenize(text.lower(), language='portuguese')
    words = [w for w in words if (w not in STOP_WORDS) or (w in IMPORTANT_WORDS)]
    words = [w for w in words if CHAR_PATTERN.search(w)]
    words = [w for w in words if len(w) > 1 or w in SINGLE_KEEP]
    return words


def train_model(pretrained_path, corpus_path, save_path, window, min_count, sg):
    pretrained = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [preprocess(line) for line in text.splitlines() if line.strip()]
    sentences = [s for s in sentences if len(s) > 3]
    model = Word2Vec(
        sentences,
        vector_size=pretrained.vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        negative=15,
        alpha=0.025,
        min_alpha=0.0001,
    )
    overlap = list(set(pretrained.index_to_key) & set(model.wv.index_to_key))
    for word in overlap:
        model.wv.vectors[model.wv.key_to_index[word]] = pretrained[word]
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)
    total_epochs = 10
    for epoch in range(total_epochs):
        alpha = model.alpha - (model.alpha - model.min_alpha) * (epoch / total_epochs)
        model.alpha = alpha
        model.train(sentences, total_examples=len(sentences), epochs=1, compute_loss=True)
    model.wv.save(save_path)
    return model.wv


def discursive_path(model, word1, word2):
    vector_dim = model.vector_size
    vec1 = model[word1]
    vec2 = model[word2]
    mean_vector = np.mean(model.vectors, axis=0)
    centered = model.vectors - mean_vector
    pca_full = PCA(n_components=vector_dim, svd_solver='randomized', random_state=42)
    pca_full.fit(centered)
    Vt = pca_full.components_
    persistence = {}
    for dim in range(vector_dim, 1, -1):
        proj = Vt[:dim]
        vec1_r = np.dot(vec1 - mean_vector, proj.T)
        vec2_r = np.dot(vec2 - mean_vector, proj.T)
        all_r = np.dot(centered, proj.T)
        midpoint = (vec1_r + vec2_r) / 2
        dists = np.linalg.norm(all_r - midpoint, axis=1)
        for idx in np.argsort(dists)[:10]:
            word = model.index_to_key[idx]
            if word in (word1, word2):
                continue
            vec = all_r[idx]
            path_vec = vec2_r - vec1_r
            path_len = np.linalg.norm(path_vec)
            path_dir = path_vec / path_len
            proj_len = np.dot(vec - vec1_r, path_dir)
            if 0 < proj_len < path_len:
                persistence[word] = persistence.get(word, 0) + 1
    min_p = vector_dim * 0.3
    robust = [w for w, c in persistence.items() if c > min_p]
    positions = {}
    for word in robust:
        vec = model[word] - mean_vector
        total = 0
        count = 0
        for dim in range(vector_dim, 50, -50):
            proj = Vt[:dim]
            v = np.dot(vec, proj.T)
            v1 = np.dot(vec1 - mean_vector, proj.T)
            v2 = np.dot(vec2 - mean_vector, proj.T)
            pv = v2 - v1
            pd = pv / np.linalg.norm(pv)
            total += np.dot(v - v1, pd)
            count += 1
        positions[word] = total / count
    ordered = sorted(positions.items(), key=lambda x: x[1])
    return [word1] + [w for w, _ in ordered] + [word2]


def plot_comparison(model_pop, model_inst, path_pop, path_inst, word1, word2):
    os.makedirs('img', exist_ok=True)
    vectors_pop = np.array([model_pop[w] for w in path_pop])
    vectors_inst = np.array([model_inst[w] for w in path_inst])
    mean_pop = np.mean(vectors_pop, axis=0)
    mean_inst = np.mean(vectors_inst, axis=0)
    pca = PCA(n_components=2)
    all_v = np.vstack((vectors_pop - mean_pop, vectors_inst - mean_inst))
    all_2d = pca.fit_transform(all_v)
    pop_2d = all_2d[:len(vectors_pop)]
    inst_2d = all_2d[len(vectors_pop):]
    plt.figure(figsize=(12, 8))
    plt.plot(pop_2d[:, 0], pop_2d[:, 1], 'k-', label='Discurso Popular', linewidth=2, alpha=0.8)
    plt.plot(inst_2d[:, 0], inst_2d[:, 1], 'k--', label='Discurso Institucional', linewidth=2, alpha=0.8)
    for i, word in enumerate(path_pop):
        plt.annotate(word, (pop_2d[i, 0], pop_2d[i, 1]), color='k', fontsize=10)
    for i, word in enumerate(path_inst):
        plt.annotate(word, (inst_2d[i, 0], inst_2d[i, 1]), color='k', fontsize=10)
    plt.title('Comparação de Caminhos Discursivos')
    plt.legend()
    plt.grid(True)
    filename = f'img/{word1}_{word2}_discursive_paths_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ensure_fasttext()
    os.makedirs('models', exist_ok=True)
    pretrained_path = FASTTEXT_VEC

    print('Treinando modelo debates...')
    model_debates = train_model(
        pretrained_path=pretrained_path,
        corpus_path='corpus_debates.txt',
        save_path='models/debates.model',
        window=7,
        min_count=2,
        sg=0,
    )

    print('Treinando modelo comentários...')
    model_comentarios = train_model(
        pretrained_path=pretrained_path,
        corpus_path='corpus_comentarios.txt',
        save_path='models/comentarios.model',
        window=10,
        min_count=8,
        sg=1,
    )

    for word1, word2 in WORD_PAIRS:
        print(f'Calculando caminho: {word1} → {word2}')
        path_pop = discursive_path(model_comentarios, word1, word2)
        path_inst = discursive_path(model_debates, word1, word2)
        print(f'  Popular: {" → ".join(path_pop)}')
        print(f'  Debate:  {" → ".join(path_inst)}')
        plot_comparison(model_comentarios, model_debates, path_pop, path_inst, word1, word2)


if __name__ == '__main__':
    main()
