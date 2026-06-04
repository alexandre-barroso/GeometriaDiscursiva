import os
import sys

# -----------------------------------------------------------------------------
# Deterministic execution guard
# -----------------------------------------------------------------------------
# Gensim's Word2Vec can vary across runs unless Python hashing, worker count, and
# numeric thread scheduling are controlled. These variables must be set before
# importing numpy/gensim, so the script re-executes itself once when needed.
DETERMINISTIC_SEED = 42
_REQUIRED_ENV = {
    'PYTHONHASHSEED': str(DETERMINISTIC_SEED),
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}

_reexec_needed = False
for _key, _value in _REQUIRED_ENV.items():
    if os.environ.get(_key) != _value:
        os.environ[_key] = _value
        if _key == 'PYTHONHASHSEED':
            _reexec_needed = True

if _reexec_needed and os.environ.get('GEOMETRIAS_DETERMINISTIC_REEXEC') != '1':
    os.environ['GEOMETRIAS_DETERMINISTIC_REEXEC'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import gzip
import hashlib
import json
import random
import re
import shutil
import urllib.request

import matplotlib.pyplot as plt
import nltk
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA

random.seed(DETERMINISTIC_SEED)
np.random.seed(DETERMINISTIC_SEED)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

FASTTEXT_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz'

# Resolve paths from the directory where this script lives, not from whatever
# directory the shell happens to use as the current working directory.
# Expected repository layout:
#   repo/
#     geometrias_discursivas_v3.py
#     corpora/
#       corpus_debates.txt
#       corpus_comentarios.txt
#     img/
#     models/
#     pretrained_models/
#     legacy/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPORA_DIR = os.path.join(BASE_DIR, 'corpora')
IMG_DIR = os.path.join(BASE_DIR, 'img')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PRE_TRAINED_MODELS = os.path.join(BASE_DIR, 'pretrained_models')

FASTTEXT_GZ = os.path.join(PRE_TRAINED_MODELS, 'cc.pt.300.vec.gz')
FASTTEXT_VEC = os.path.join(PRE_TRAINED_MODELS, 'cc.pt.300.vec')

# Output is now vector PDF, not PNG. The filename stem is unchanged.
OUTPUT_FORMAT = 'pdf'
DELETE_STALE_PNG_FILES = True

# Reproducibility / paper-lock options. Defaults are conservative: do not
# overwrite existing models, and reuse the saved path manifest once it exists.
FORCE_RETRAIN_MODELS = os.environ.get('GEOMETRIAS_FORCE_RETRAIN', '0') == '1'
REBUILD_PATH_MANIFEST = os.environ.get('GEOMETRIAS_REBUILD_PATHS', '0') == '1'
PATH_MANIFEST = os.path.join(MODELS_DIR, 'discursive_paths_manifest.json')

# Larger picture-in-picture panel + tighter crop around the dense core.
ZOOM_CORE_FRACTION = 0.58
ZOOM_MIN_CORE_POINTS = 8
ZOOM_MAX_FRACTION_OF_FULL_WIDTH = 0.46
ZOOM_MAX_FRACTION_OF_FULL_HEIGHT = 0.46
INSET_WIDTH = 0.60
INSET_HEIGHT = 0.58

WORD_PAIRS = [
    ('democracia', 'ditadura'),
    ('dinheiro', 'elite'),
    ('governo', 'população'),
    ('pobres', 'elite'),
    ('população', 'desemprego'),
]

STOP_WORDS = set(stopwords.words('portuguese'))
IMPORTANT_WORDS = {'não', 'sim', 'muito', 'pouco', 'mais', 'menos'}
CHAR_PATTERN = re.compile(r'[a-záéíóúâêîôûãõçàèìòùäëïöüñ]+')
SINGLE_KEEP = {'é', 'à', 'e', 'o', 'a'}


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

plt.rcParams.update({
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
})


def stable_hash(text):
    """Deterministic 32-bit hash for gensim initial-vector seeding."""
    if isinstance(text, bytes):
        data = text
    else:
        data = str(text).encode('utf-8')
    return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16) & 0xFFFFFFFF


def preprocess(text):
    words = word_tokenize(text.lower(), language='portuguese')
    words = [w for w in words if (w not in STOP_WORDS) or (w in IMPORTANT_WORDS)]
    words = [w for w in words if CHAR_PATTERN.search(w)]
    words = [w for w in words if len(w) > 1 or w in SINGLE_KEEP]
    return words


def _load_path_manifest():
    if not os.path.exists(PATH_MANIFEST):
        return {}
    try:
        with open(PATH_MANIFEST, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_path_manifest(manifest):
    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp_path = PATH_MANIFEST + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp_path, PATH_MANIFEST)


def _manifest_key(word1, word2):
    return f'{word1}::{word2}'


def _path_exists_in_model(model, path):
    if not isinstance(path, list) or not path:
        return False
    return all(isinstance(word, str) and word in model for word in path)


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
        workers=1,
        sg=sg,
        negative=15,
        alpha=0.025,
        min_alpha=0.0001,
        seed=DETERMINISTIC_SEED,
        hashfxn=stable_hash,
        sorted_vocab=1,
        epochs=5,
    )

    overlap = sorted(set(pretrained.index_to_key) & set(model.wv.index_to_key))
    for word in overlap:
        model.wv.vectors[model.wv.key_to_index[word]] = pretrained[word]

    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)

    total_epochs = 10
    for epoch in range(total_epochs):
        alpha = model.alpha - (model.alpha - model.min_alpha) * (epoch / total_epochs)
        model.alpha = alpha
        model.train(sentences, total_examples=len(sentences), epochs=1, compute_loss=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.wv.save(save_path)
    return model.wv


def load_or_train_model(pretrained_path, corpus_path, save_path, window, min_count, sg, label):
    if os.path.exists(save_path) and not FORCE_RETRAIN_MODELS:
        print(f'Carregando modelo {label}: {save_path}')
        return KeyedVectors.load(save_path, mmap=None)

    if FORCE_RETRAIN_MODELS and os.path.exists(save_path):
        print(f'Retreinando modelo {label} de forma determinística: {save_path}')
    else:
        print(f'Treinando modelo {label} de forma determinística...')

    return train_model(
        pretrained_path=pretrained_path,
        corpus_path=corpus_path,
        save_path=save_path,
        window=window,
        min_count=min_count,
        sg=sg,
    )


def discursive_path(model, word1, word2):
    vector_dim = model.vector_size
    vec1 = model[word1]
    vec2 = model[word2]

    mean_vector = np.mean(model.vectors, axis=0)
    centered = model.vectors - mean_vector

    n_components = min(vector_dim, centered.shape[0], centered.shape[1])
    pca_full = PCA(n_components=n_components, svd_solver='full')
    pca_full.fit(centered)
    Vt = pca_full.components_

    persistence = {}
    for dim in range(n_components, 1, -1):
        proj = Vt[:dim]
        vec1_r = np.dot(vec1 - mean_vector, proj.T)
        vec2_r = np.dot(vec2 - mean_vector, proj.T)
        all_r = np.dot(centered, proj.T)

        midpoint = (vec1_r + vec2_r) / 2
        dists = np.linalg.norm(all_r - midpoint, axis=1)

        ranked_indices = sorted(
            range(len(dists)),
            key=lambda idx: (float(dists[idx]), model.index_to_key[idx]),
        )

        for idx in ranked_indices[:10]:
            word = model.index_to_key[idx]
            if word in (word1, word2):
                continue

            vec = all_r[idx]
            path_vec = vec2_r - vec1_r
            path_len = np.linalg.norm(path_vec)
            if path_len < 1e-12:
                continue

            path_dir = path_vec / path_len
            proj_len = np.dot(vec - vec1_r, path_dir)
            if 0 < proj_len < path_len:
                persistence[word] = persistence.get(word, 0) + 1

    min_p = n_components * 0.3
    robust = sorted(w for w, c in persistence.items() if c > min_p)

    positions = {}
    for word in robust:
        vec = model[word] - mean_vector
        total = 0.0
        count = 0

        for dim in range(n_components, 50, -50):
            proj = Vt[:dim]
            v = np.dot(vec, proj.T)
            v1 = np.dot(vec1 - mean_vector, proj.T)
            v2 = np.dot(vec2 - mean_vector, proj.T)
            pv = v2 - v1
            pv_norm = np.linalg.norm(pv)
            if pv_norm < 1e-12:
                continue
            pd = pv / pv_norm
            total += float(np.dot(v - v1, pd))
            count += 1

        if count:
            positions[word] = total / count

    ordered = sorted(positions.items(), key=lambda item: (round(float(item[1]), 12), item[0]))
    return [word1] + [w for w, _ in ordered] + [word2]


def compute_or_load_paths(model_pop, model_inst, word1, word2):
    manifest = _load_path_manifest()
    key = _manifest_key(word1, word2)

    if not REBUILD_PATH_MANIFEST and key in manifest:
        entry = manifest[key]
        path_pop = entry.get('popular')
        path_inst = entry.get('institucional')
        if _path_exists_in_model(model_pop, path_pop) and _path_exists_in_model(model_inst, path_inst):
            print('  Usando caminhos congelados no manifesto.')
            return path_pop, path_inst
        print('  Manifesto encontrado, mas incompatível com os modelos atuais; recalculando.')

    path_pop = discursive_path(model_pop, word1, word2)
    path_inst = discursive_path(model_inst, word1, word2)

    manifest[key] = {
        'word1': word1,
        'word2': word2,
        'seed': DETERMINISTIC_SEED,
        'popular': path_pop,
        'institucional': path_inst,
    }
    _save_path_manifest(manifest)
    return path_pop, path_inst


def plot_comparison(model_pop, model_inst, path_pop, path_inst, word1, word2):
    """Plot full discursive paths plus a large picture-in-picture zoom.

    The geometry is not recomputed inside the inset. The inset is a crop of the
    same PCA coordinate system shown on the macro plot.
    """
    from matplotlib.patches import Rectangle

    os.makedirs(IMG_DIR, exist_ok=True)

    vectors_pop = np.array([model_pop[w] for w in path_pop])
    vectors_inst = np.array([model_inst[w] for w in path_inst])

    mean_pop = np.mean(vectors_pop, axis=0)
    mean_inst = np.mean(vectors_inst, axis=0)

    pca = PCA(n_components=2, svd_solver='full')
    all_v = np.vstack((vectors_pop - mean_pop, vectors_inst - mean_inst))
    all_2d = pca.fit_transform(all_v)

    pop_2d = all_2d[:len(vectors_pop)]
    inst_2d = all_2d[len(vectors_pop):]
    all_points = np.vstack((pop_2d, inst_2d))

    def _finite_range(values):
        low = float(np.min(values))
        high = float(np.max(values))
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = -1.0, 1.0
        if abs(high - low) < 1e-9:
            low -= 0.5
            high += 0.5
        return low, high

    def _limits(points, pad_fraction=0.12):
        xmin, xmax = _finite_range(points[:, 0])
        ymin, ymax = _finite_range(points[:, 1])
        xpad = max((xmax - xmin) * pad_fraction, 0.08)
        ypad = max((ymax - ymin) * pad_fraction, 0.08)
        return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

    def _automatic_zoom_box(points):
        # Crop the dense central mass. The crop is intentionally tighter than the
        # previous version, while the inset panel itself is much larger.
        if len(points) <= ZOOM_MIN_CORE_POINTS:
            core = points
        else:
            center = np.median(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            n_core = int(np.ceil(len(points) * ZOOM_CORE_FRACTION))
            n_core = max(ZOOM_MIN_CORE_POINTS, min(len(points), n_core))
            core = points[np.argsort(distances, kind='mergesort')[:n_core]]

        xmin, xmax = _finite_range(core[:, 0])
        ymin, ymax = _finite_range(core[:, 1])

        full_xmin, full_xmax, full_ymin, full_ymax = _limits(points, pad_fraction=0.02)
        full_width = full_xmax - full_xmin
        full_height = full_ymax - full_ymin

        width = xmax - xmin
        height = ymax - ymin
        xpad = max(width * 0.20, full_width * 0.020, 0.035)
        ypad = max(height * 0.24, full_height * 0.020, 0.035)

        xmin -= xpad
        xmax += xpad
        ymin -= ypad
        ymax += ypad

        max_zoom_width = full_width * ZOOM_MAX_FRACTION_OF_FULL_WIDTH
        max_zoom_height = full_height * ZOOM_MAX_FRACTION_OF_FULL_HEIGHT
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2

        if xmax - xmin > max_zoom_width:
            xmin = center_x - max_zoom_width / 2
            xmax = center_x + max_zoom_width / 2
        if ymax - ymin > max_zoom_height:
            ymin = center_y - max_zoom_height / 2
            ymax = center_y + max_zoom_height / 2

        return xmin, xmax, ymin, ymax

    def _inside_box(points, box):
        xmin, xmax, ymin, ymax = box
        return (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
        )

    def _choose_inset_position(points, xlim, ylim):
        candidates = [
            (1.0 - INSET_WIDTH - 0.025, 1.0 - INSET_HEIGHT - 0.045, INSET_WIDTH, INSET_HEIGHT),
            (0.025, 1.0 - INSET_HEIGHT - 0.045, INSET_WIDTH, INSET_HEIGHT),
            (1.0 - INSET_WIDTH - 0.025, 0.040, INSET_WIDTH, INSET_HEIGHT),
            (0.025, 0.040, INSET_WIDTH, INSET_HEIGHT),
        ]

        xmin, xmax = xlim
        ymin, ymax = ylim
        xr = xmax - xmin
        yr = ymax - ymin

        best_score = None
        best_box = candidates[0]
        for box in candidates:
            bx, by, bw, bh = box
            data_x0 = xmin + bx * xr
            data_x1 = xmin + (bx + bw) * xr
            data_y0 = ymin + by * yr
            data_y1 = ymin + (by + bh) * yr
            covered = (
                (points[:, 0] >= data_x0) & (points[:, 0] <= data_x1) &
                (points[:, 1] >= data_y0) & (points[:, 1] <= data_y1)
            )
            # Prefer locations that cover fewer outlying points. If tied, upper
            # positions are slightly preferred because they read more like PIP.
            score = int(np.sum(covered)) - (0.25 if by > 0.50 else 0.0)
            if best_score is None or score < best_score:
                best_score = score
                best_box = box
        return best_box

    def _plot_path(ax, pts, linestyle, marker, label=None, linewidth=2.0, alpha=0.86, arrows=True):
        ax.plot(
            pts[:, 0], pts[:, 1],
            linestyle=linestyle,
            color='0.16',
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
            markersize=4.2,
            markerfacecolor='white',
            markeredgecolor='0.16',
            markeredgewidth=0.85,
            label=label,
            zorder=3,
        )

        if not arrows:
            return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xr = xlim[1] - xlim[0]
        yr = ylim[1] - ylim[0]
        xmargin = xr * 0.015
        ymargin = yr * 0.015

        def _inside_visible(p):
            return (
                xlim[0] - xmargin <= p[0] <= xlim[1] + xmargin and
                ylim[0] - ymargin <= p[1] <= ylim[1] + ymargin
            )

        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) < 1e-9:
                continue
            if not (_inside_visible(a) and _inside_visible(b)):
                continue
            ax.annotate(
                '',
                xy=(b[0], b[1]),
                xytext=(a[0], a[1]),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='0.16',
                    lw=max(0.55, linewidth * 0.36),
                    linestyle=linestyle,
                    alpha=min(0.62, alpha),
                    shrinkA=6,
                    shrinkB=6,
                    mutation_scale=7.5,
                ),
                annotation_clip=True,
                zorder=2,
            )

    def _label_positions(ax, anchors, labels, fontsize, mode):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xr = xlim[1] - xlim[0]
        yr = ylim[1] - ylim[0]

        anchors_ax = np.column_stack((
            (anchors[:, 0] - xlim[0]) / xr,
            (anchors[:, 1] - ylim[0]) / yr,
        ))
        anchors_ax = np.clip(anchors_ax, 0.01, 0.99)

        center = np.mean(anchors_ax, axis=0)
        delta = anchors_ax - center
        dist = np.linalg.norm(delta, axis=1)
        angles = np.arctan2(delta[:, 1], delta[:, 0])

        if np.nanmax(dist) < 0.06:
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

        unit = np.column_stack((np.cos(angles), np.sin(angles)))
        unit[~np.isfinite(unit)] = 0.0

        if mode == 'inset':
            base_offset = 0.125 if len(labels) <= 18 else 0.155
            preferred = anchors_ax + unit * base_offset
        else:
            preferred = anchors_ax + unit * 0.045

        positions = preferred.copy()

        widths = np.array([0.024 + 0.0057 * min(len(str(s)), 20) for s in labels])
        heights = np.full(len(labels), 0.033 if fontsize <= 8 else 0.042)
        if mode == 'inset':
            heights *= 1.03

        # Deterministic mini force layout in axis coordinates. It does not move
        # data points; it only moves the labels and draws leader lines.
        for _ in range(320):
            movement = (preferred - positions) * 0.013
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    overlap_x = (widths[i] + widths[j]) / 2 + 0.010 - abs(dx)
                    overlap_y = (heights[i] + heights[j]) / 2 + 0.008 - abs(dy)
                    if overlap_x > 0 and overlap_y > 0:
                        if overlap_x < overlap_y:
                            push = overlap_x / 2 + 0.002
                            direction = 1.0 if dx >= 0 else -1.0
                            movement[i, 0] -= push * direction
                            movement[j, 0] += push * direction
                        else:
                            push = overlap_y / 2 + 0.002
                            direction = 1.0 if dy >= 0 else -1.0
                            movement[i, 1] -= push * direction
                            movement[j, 1] += push * direction

            positions += movement
            positions[:, 0] = np.clip(positions[:, 0], widths / 2 + 0.010, 1 - widths / 2 - 0.010)
            positions[:, 1] = np.clip(positions[:, 1], heights / 2 + 0.012, 1 - heights / 2 - 0.012)

        return np.column_stack((xlim[0] + positions[:, 0] * xr, ylim[0] + positions[:, 1] * yr))

    def _annotate_words(ax, pts, words, fontsize=8, mode='main'):
        if len(words) == 0:
            return

        pts = np.asarray(pts, dtype=float)
        text_xy = _label_positions(ax, pts, words, fontsize=fontsize, mode=mode)

        for point, label_xy, word in zip(pts, text_xy, words):
            ax.annotate(
                str(word),
                xy=(point[0], point[1]),
                xytext=(label_xy[0], label_xy[1]),
                textcoords='data',
                ha='center',
                va='center',
                fontsize=fontsize,
                color='0.05',
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.88),
                arrowprops=dict(
                    arrowstyle='-',
                    color='0.30',
                    lw=0.55,
                    alpha=0.78,
                    shrinkA=2,
                    shrinkB=3,
                ),
                annotation_clip=True,
                zorder=6,
            )

    fig, ax = plt.subplots(figsize=(17.0, 10.0))
    fig.patch.set_facecolor('white')

    full_limits = _limits(all_points, pad_fraction=0.14)
    full_xlim = full_limits[:2]
    full_ylim_tuple = full_limits[2:]
    ax.set_xlim(*full_xlim)
    ax.set_ylim(*full_ylim_tuple)

    _plot_path(ax, pop_2d, '-', 'o', label='Discurso Popular', linewidth=2.35, alpha=0.88, arrows=True)
    _plot_path(ax, inst_2d, '--', 's', label='Discurso Institucional', linewidth=2.35, alpha=0.88, arrows=True)

    zoom_box = _automatic_zoom_box(all_points)
    zx0, zx1, zy0, zy1 = zoom_box

    zoom_rect = Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        fill=False,
        edgecolor='0.20',
        linewidth=1.05,
        linestyle=':',
        zorder=4,
    )
    ax.add_patch(zoom_rect)
    ax.text(
        zx0,
        zy1,
        ' detalhe ampliado ',
        fontsize=8.5,
        ha='left',
        va='bottom',
        color='0.20',
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='0.75', alpha=0.88),
        zorder=7,
    )

    pop_in_zoom = _inside_box(pop_2d, zoom_box)
    inst_in_zoom = _inside_box(inst_2d, zoom_box)

    main_pop_idx = [i for i in range(len(path_pop)) if i in (0, len(path_pop) - 1) or not pop_in_zoom[i]]
    main_inst_idx = [i for i in range(len(path_inst)) if i in (0, len(path_inst) - 1) or not inst_in_zoom[i]]

    _annotate_words(ax, pop_2d[main_pop_idx], [path_pop[i] for i in main_pop_idx], fontsize=9, mode='main')
    _annotate_words(ax, inst_2d[main_inst_idx], [path_inst[i] for i in main_inst_idx], fontsize=9, mode='main')

    ax.set_title(f'Comparação de Caminhos Discursivos: {word1} → {word2}', fontsize=16, pad=16)
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.axhline(0, color='0.80', lw=0.7, zorder=0)
    ax.axvline(0, color='0.80', lw=0.7, zorder=0)
    ax.set_aspect('equal', adjustable='box')

    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        fontsize=10,
    )
    legend.get_frame().set_edgecolor('0.78')

    inset_box_axes = _choose_inset_position(all_points, ax.get_xlim(), ax.get_ylim())
    axins = ax.inset_axes(inset_box_axes)
    axins.set_facecolor('white')
    for spine in axins.spines.values():
        spine.set_linewidth(1.05)
        spine.set_edgecolor('0.20')

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)

    _plot_path(axins, pop_2d, '-', 'o', label=None, linewidth=1.72, alpha=0.90, arrows=True)
    _plot_path(axins, inst_2d, '--', 's', label=None, linewidth=1.72, alpha=0.90, arrows=True)
    axins.set_aspect('equal', adjustable='box')
    axins.grid(True, alpha=0.22, linewidth=0.6)
    axins.tick_params(labelsize=7, length=2.5, pad=1)
    axins.set_title('zoom da zona densa', fontsize=10, pad=4)

    inset_pop_idx = [i for i, inside in enumerate(pop_in_zoom) if inside]
    inset_inst_idx = [i for i, inside in enumerate(inst_in_zoom) if inside]

    _annotate_words(axins, pop_2d[inset_pop_idx], [path_pop[i] for i in inset_pop_idx], fontsize=7.7, mode='inset')
    _annotate_words(axins, inst_2d[inset_inst_idx], [path_inst[i] for i in inset_inst_idx], fontsize=7.7, mode='inset')

    try:
        ax.indicate_inset_zoom(axins, edgecolor='0.25', linewidth=0.7, alpha=0.8)
    except Exception:
        # Older Matplotlib versions may not have indicate_inset_zoom. The dotted
        # rectangle still marks the zoomed area, so plotting can continue.
        pass

    filename = os.path.join(IMG_DIR, f'{word1}_{word2}_discursive_paths_comparison.{OUTPUT_FORMAT}')
    fig.savefig(
        filename,
        bbox_inches='tight',
        format=OUTPUT_FORMAT,
        metadata={
            'Creator': 'geometrias_discursivas_inset_zoom.py',
            'Producer': 'Matplotlib',
            'CreationDate': None,
            'ModDate': None,
        },
    )
    plt.close(fig)

    if DELETE_STALE_PNG_FILES:
        stale_png = os.path.join(IMG_DIR, f'{word1}_{word2}_discursive_paths_comparison.png')
        if os.path.exists(stale_png):
            os.remove(stale_png)

    print(f'  Figura salva: {filename}')


def main():
    ensure_fasttext()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    pretrained_path = FASTTEXT_VEC

    model_debates = load_or_train_model(
        pretrained_path=pretrained_path,
        corpus_path=os.path.join(CORPORA_DIR, 'corpus_debates.txt'),
        save_path=os.path.join(MODELS_DIR, 'debates.model'),
        window=7,
        min_count=2,
        sg=0,
        label='debates',
    )

    model_comentarios = load_or_train_model(
        pretrained_path=pretrained_path,
        corpus_path=os.path.join(CORPORA_DIR, 'corpus_comentarios.txt'),
        save_path=os.path.join(MODELS_DIR, 'comentarios.model'),
        window=10,
        min_count=8,
        sg=1,
        label='comentários',
    )

    for word1, word2 in WORD_PAIRS:
        print(f'Calculando caminho: {word1} → {word2}')
        path_pop, path_inst = compute_or_load_paths(model_comentarios, model_debates, word1, word2)
        print(f'  Popular: {" → ".join(path_pop)}')
        print(f'  Debate:  {" → ".join(path_inst)}')
        plot_comparison(model_comentarios, model_debates, path_pop, path_inst, word1, word2)

    print(f'Manifesto de caminhos: {PATH_MANIFEST}')
    print('Saídas em PDF gravadas na pasta img/.')


if __name__ == '__main__':
    main()
