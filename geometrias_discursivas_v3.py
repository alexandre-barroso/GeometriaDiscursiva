import os
import sys


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
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPORA_DIR = os.path.join(BASE_DIR, 'corpora')
IMG_DIR = os.path.join(BASE_DIR, 'img')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PRE_TRAINED_MODELS = os.path.join(BASE_DIR, 'pretrained_models')

FASTTEXT_GZ = os.path.join(PRE_TRAINED_MODELS, 'cc.pt.300.vec.gz')
FASTTEXT_VEC = os.path.join(PRE_TRAINED_MODELS, 'cc.pt.300.vec')


OUTPUT_FORMAT = 'pdf'
DELETE_STALE_PNG_FILES = True


FORCE_RETRAIN_MODELS = os.environ.get('GEOMETRIAS_FORCE_RETRAIN', '0') == '1'
REBUILD_PATH_MANIFEST = os.environ.get('GEOMETRIAS_REBUILD_PATHS', '0') == '1'
PATH_MANIFEST = os.path.join(MODELS_DIR, 'discursive_paths_manifest.json')


ZOOM_CORE_FRACTION = 0.58
ZOOM_MIN_CORE_POINTS = 8
ZOOM_MAX_FRACTION_OF_FULL_WIDTH = 0.46
ZOOM_MAX_FRACTION_OF_FULL_HEIGHT = 0.46
FIGURE_SIZE = (8.0, 8.0)
LABEL_FONT_SIZE = 11.5
LABEL_PADDING_PT = 2.5
LABEL_GAP_PT = 3.0

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


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'figure.dpi': 144,
    'axes.linewidth': 0.65,
    'axes.edgecolor': '0.45',
    'axes.grid': True,
    'grid.color': '0.90',
    'grid.linewidth': 0.45,
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
})


def stable_hash(text):

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


def _finite_range(values):
    low, high = float(np.min(values)), float(np.max(values))
    if not np.isfinite([low, high]).all():
        raise ValueError('A projeção contém coordenadas não finitas.')
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    return low, high


def _limits(points, pad_fraction=0.14):
    xmin, xmax = _finite_range(points[:, 0])
    ymin, ymax = _finite_range(points[:, 1])
    xpad = max((xmax - xmin) * pad_fraction, 0.08)
    ypad = max((ymax - ymin) * pad_fraction, 0.08)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _automatic_zoom_box(points):

    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    n_core = min(len(points), max(ZOOM_MIN_CORE_POINTS,
                                 int(np.ceil(len(points) * ZOOM_CORE_FRACTION))))
    core = points[np.argsort(distances, kind='mergesort')[:n_core]]
    xmin, xmax = _finite_range(core[:, 0])
    ymin, ymax = _finite_range(core[:, 1])
    fx0, fx1, fy0, fy1 = _limits(points, pad_fraction=0.02)
    fw, fh = fx1 - fx0, fy1 - fy0
    width = min(xmax - xmin + 2 * max((xmax - xmin) * 0.20, fw * 0.02, 0.035),
                fw * ZOOM_MAX_FRACTION_OF_FULL_WIDTH)
    height = min(ymax - ymin + 2 * max((ymax - ymin) * 0.24, fh * 0.02, 0.035),
                 fh * ZOOM_MAX_FRACTION_OF_FULL_HEIGHT)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2


def _equal_scale_limits(ax, limits):

    xmin, xmax, ymin, ymax = limits
    ratio = ax.get_window_extent().width / ax.get_window_extent().height
    width, height = xmax - xmin, ymax - ymin
    width, height = max(width, height * ratio), max(height, width / ratio)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(cx - width / 2, cx + width / 2)
    ax.set_ylim(cy - height / 2, cy + height / 2)
    ax.set_aspect('equal', adjustable='box')
    return (*ax.get_xlim(), *ax.get_ylim())


def _inside_box(points, box):
    xmin, xmax, ymin, ymax = box
    return ((points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax))


def _zoom_with_clear_markers(ax, points):

    box = _equal_scale_limits(ax, _automatic_zoom_box(points))
    for _ in range(len(points)):
        xmin, xmax, ymin, ymax = box
        dx, dy = (xmax - xmin) * 0.025, (ymax - ymin) * 0.04
        nearby = points[_inside_box(points, (xmin - dx, xmax + dx, ymin - dy, ymax + dy))]
        if not len(nearby):
            break
        expanded = (min(xmin, float(nearby[:, 0].min()) - dx),
                    max(xmax, float(nearby[:, 0].max()) + dx),
                    min(ymin, float(nearby[:, 1].min()) - dy),
                    max(ymax, float(nearby[:, 1].max()) + dy))
        if np.allclose(expanded, box, rtol=0, atol=1e-10):
            break
        box = _equal_scale_limits(ax, expanded)
    return box


def _segments_hit_boxes(starts, ends, boxes):

    starts = np.asarray(starts).reshape(-1, 2)
    ends = np.asarray(ends).reshape(-1, 2)
    boxes = np.asarray(boxes).reshape(-1, 4)
    origin, delta = starts[:, None, :], (ends - starts)[:, None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (boxes[None, :, :2] - origin) / delta
        t1 = (boxes[None, :, 2:] - origin) / delta
    parallel = np.abs(delta) < 1e-10
    outside = (origin < boxes[None, :, :2]) | (origin > boxes[None, :, 2:])
    lower = np.where(parallel, -np.inf, np.minimum(t0, t1))
    upper = np.where(parallel, np.inf, np.maximum(t0, t1))
    return ((np.maximum(lower.max(axis=2), 0) <= np.minimum(upper.min(axis=2), 1)) &
            ~np.any(parallel & outside, axis=2))


def _segments_cross(a, b, c, d):
    a, b = np.asarray(a)[:, None, :], np.asarray(b)[:, None, :]
    c, d = np.asarray(c)[None, :, :], np.asarray(d)[None, :, :]
    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return ((cross(b - a, c - a) * cross(b - a, d - a) < -1e-6) &
            (cross(d - c, a - c) * cross(d - c, b - c) < -1e-6))


def _place_labels(ax, records, all_points, segments, reserved=()):


    if not records:
        return []
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    unit = fig.dpi / 72
    gap = LABEL_GAP_PT * unit
    padding = LABEL_PADDING_PT * unit
    bounds = ax.get_window_extent().padded(-4 * unit)
    nodes = ax.transData.transform(all_points)
    visible = _inside_box(all_points, (*ax.get_xlim(), *ax.get_ylim()))
    nodes = nodes[visible]
    node_boxes = np.column_stack((nodes - 3.2 * unit, nodes + 3.2 * unit))
    starts = ax.transData.transform(segments[:, 0])
    ends = ax.transData.transform(segments[:, 1])
    obstacles = np.asarray(reserved).reshape(-1, 4)

    artists, anchors, sizes = [], [], []
    for record in records:
        text = ax.text(*record['point'], record['word'], ha='center', va='center',
                       fontsize=LABEL_FONT_SIZE,
                       fontweight='bold' if record['endpoint'] else 'normal',
                       color='0.08', zorder=6,
                       bbox=dict(boxstyle='square,pad=0.12', fc='white', ec='none'))
        extent = text.get_window_extent(renderer)
        artists.append(text)
        anchors.append(ax.transData.transform(record['point']))
        sizes.append([extent.width + 2 * padding, extent.height + 2 * padding])
    anchors, sizes = np.array(anchors), np.array(sizes)

    angles = np.arange(48) * (2 * np.pi / 48)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates = []
    for anchor, size in zip(anchors, sizes):
        offsets = np.concatenate([
            directions * (size / 2 + (4 + radius) * unit)
            for radius in (0, 4, 9, 16, 25, 37, 52, 72, 96, 125, 160)
        ])
        centers = anchor + offsets
        boxes = np.column_stack((centers - size / 2, centers + size / 2))
        valid = ((boxes[:, 0] >= bounds.x0) & (boxes[:, 1] >= bounds.y0) &
                 (boxes[:, 2] <= bounds.x1) & (boxes[:, 3] <= bounds.y1))
        for fixed in (node_boxes, obstacles):
            overlaps = ((boxes[:, None, 0] < fixed[None, :, 2]) &
                        (boxes[:, None, 2] > fixed[None, :, 0]) &
                        (boxes[:, None, 1] < fixed[None, :, 3]) &
                        (boxes[:, None, 3] > fixed[None, :, 1]))
            valid &= ~overlaps.any(axis=1)
        centers, boxes = centers[valid], boxes[valid]
        delta = anchor - centers
        edge_fraction = 1 / np.max(np.abs(delta) / (size / 2), axis=1)
        leader_start = centers + delta * edge_fraction[:, None]
        length = np.linalg.norm(anchor - leader_start, axis=1)
        direction = (anchor - leader_start) / np.maximum(length[:, None], 1e-9)
        leader_start += direction * np.minimum(length / 3, 1 * unit)[:, None]
        leader_end = anchor - direction * np.minimum(length / 3, 3 * unit)[:, None]


        other_nodes = node_boxes[np.linalg.norm(nodes - anchor, axis=1) > 6.4 * unit]
        clear = ~_segments_hit_boxes(leader_start, leader_end, other_nodes).any(axis=1)
        clear &= ~_segments_hit_boxes(leader_start, leader_end, obstacles).any(axis=1)
        centers, boxes = centers[clear], boxes[clear]
        leader_start, leader_end, length = leader_start[clear], leader_end[clear], length[clear]
        path_hits = _segments_hit_boxes(starts, ends, boxes).sum(axis=0)
        path_crossings = _segments_cross(leader_start, leader_end, starts, ends).sum(axis=1)

        score = (path_hits * 4000 + path_crossings * 25 +
                 length / unit + np.linalg.norm(centers - anchor, axis=1) / unit * 0.1)
        candidates.append((centers, boxes, leader_start, leader_end, score))

    def choose(i, placed):
        centers, boxes, la, lb, scores = candidates[i]
        valid = np.ones(len(boxes), dtype=bool)
        cost = scores.copy()
        if placed:
            prev_boxes = np.array([item[1] for item in placed.values()])
            pa = np.array([item[2] for item in placed.values()])
            pb = np.array([item[3] for item in placed.values()])
            overlaps = ((boxes[:, None, 0] < prev_boxes[None, :, 2] + gap) &
                        (boxes[:, None, 2] > prev_boxes[None, :, 0] - gap) &
                        (boxes[:, None, 1] < prev_boxes[None, :, 3] + gap) &
                        (boxes[:, None, 3] > prev_boxes[None, :, 1] - gap))
            valid &= ~overlaps.any(axis=1)
            valid &= ~_segments_hit_boxes(la, lb, prev_boxes).any(axis=1)
            valid &= ~_segments_hit_boxes(pa, pb, boxes).any(axis=0)
            valid &= ~_segments_cross(la, lb, pa, pb).any(axis=1)
        cost[~valid] = np.inf
        if not len(cost) or not np.isfinite(cost).any():
            return None
        j = int(np.argmin(cost))
        return centers[j], boxes[j], la[j], lb[j], cost[j]


    order = sorted(range(len(records)), key=lambda i: (len(candidates[i][0]), -sizes[i, 0], i))
    best = None
    rng = np.random.default_rng(DETERMINISTIC_SEED)
    orders = [order[shift:] + order[:shift] for shift in range(len(records))]
    orders += [list(reversed(trial)) for trial in orders]
    orders += [rng.permutation(order).tolist() for _ in range(48)]
    for attempt, trial_order in enumerate(orders):
        placed = {}
        for i in trial_order:
            candidate = choose(i, placed)
            if candidate is None:
                break
            placed[i] = candidate
        if len(placed) == len(records):
            total = sum(item[4] for item in placed.values())
            if best is None or total < best[0]:
                best = total, placed

            if attempt >= 5:
                break
    if best is None:
        for artist in artists:
            artist.remove()
        raise RuntimeError('Não foi possível separar todos os rótulos; amplie o painel.')


    placed = best[1]
    for _ in range(6):
        improved = False
        for i in sorted(placed, key=lambda i: (-placed[i][4], i)):
            previous = placed.pop(i)
            candidate = choose(i, placed)
            if candidate is not None and candidate[4] < previous[4] - 1e-6:
                placed[i], improved = candidate, True
            else:
                placed[i] = previous
        if not improved:
            break

    result = []
    for i, (record, artist) in enumerate(zip(records, artists)):
        center, box, la, lb, _ = placed[i]
        artist.set_position(ax.transData.inverted().transform(center))
        leader = ax.transData.inverted().transform([la, lb])
        ax.plot(leader[:, 0], leader[:, 1], color='0.45', linewidth=0.55,
                solid_capstyle='round', zorder=4)
        result.append(dict(record, artist=artist, leader=np.array([la, lb])))
    return result


def _validate_labels(fig, ax, labels):

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item['artist'].get_window_extent(renderer).padded(
        LABEL_PADDING_PT * fig.dpi / 72) for item in labels]
    bounds = ax.get_window_extent()
    errors = []
    for i, box in enumerate(boxes):
        if not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)):
            errors.append(f"rótulo cortado: {labels[i]['word']}")
        for j in range(i):
            if box.padded(LABEL_GAP_PT * fig.dpi / 72).overlaps(boxes[j]):
                errors.append(f"{labels[i]['word']} / {labels[j]['word']}")
        for j, item in enumerate(labels):
            if i != j and _segments_hit_boxes(
                    item['leader'][:1], item['leader'][1:], [box.extents])[0, 0]:
                errors.append(f"linha de {item['word']} sobre {labels[i]['word']}")
    if errors:
        raise RuntimeError('Colisões na figura: ' + '; '.join(errors))
    return len(labels)


def _plot_path(ax, points, linestyle, marker, label):
    line, = ax.plot(points[:, 0], points[:, 1], linestyle=linestyle,
                    color='0.22', linewidth=1.15, marker=marker, markersize=4.2,
                    markerfacecolor='white', markeredgewidth=0.85,
                    label=label, zorder=3)

    visible = _inside_box(points, (*ax.get_xlim(), *ax.get_ylim()))
    display = ax.transData.transform(points)
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        if not (visible[i] and visible[i + 1]):
            continue
        if np.linalg.norm(display[i + 1] - display[i]) < 14 * ax.figure.dpi / 72:
            continue
        ax.annotate('', xy=a + 0.72 * (b - a), xytext=a + 0.54 * (b - a),
                    arrowprops=dict(arrowstyle='-|>', color='0.22', lw=0.75,
                                    shrinkA=0, shrinkB=0, mutation_scale=7),
                    zorder=3)
    return line


def plot_comparison(model_pop, model_inst, path_pop, path_inst, word1, word2):

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
    segments = np.array([(a, b) for points in (pop_2d, inst_2d)
                         for a, b in zip(points[:-1], points[1:])])

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor='white')
    ax = fig.add_axes([0.105, 0.595, 0.865, 0.265])
    detail = fig.add_axes([0.105, 0.085, 0.865, 0.405])
    fig.suptitle(f'Caminhos discursivos: {word1} → {word2}', fontsize=15, y=0.985)
    for panel, title in ((ax, '(a) Visão geral'), (detail, '(b) Ampliação da zona densa')):
        panel.set_title(title, loc='left', fontsize=11.5, pad=9)
        panel.set_xlabel('Componente principal 1', fontsize=10, labelpad=5)
        panel.set_ylabel('Componente principal 2', fontsize=10, labelpad=5)
        panel.set_axisbelow(True)
        panel.tick_params(labelsize=9, length=3, width=0.6, pad=4)
        for axis in (panel.xaxis, panel.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=5))
            axis.set_major_formatter(FuncFormatter(
                lambda value, _: f'{0 if abs(value) < 1e-10 else value:g}'.replace('.', ',')))
    fig.canvas.draw()
    _equal_scale_limits(ax, _limits(all_points, pad_fraction=0.20))
    zoom_box = _zoom_with_clear_markers(detail, all_points)
    fig.canvas.draw()

    handles = []
    for panel in (ax, detail):
        h1 = _plot_path(panel, pop_2d, '-', 'o', 'Corpus online')
        h2 = _plot_path(panel, inst_2d, '--', 's', 'Debate eleitoral')
        if panel is ax:
            handles = [h1, h2]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.948),
               ncol=2, frameon=False, fontsize=11, handlelength=3, columnspacing=3)

    zx0, zx1, zy0, zy1 = zoom_box
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                           fill=False, edgecolor='0.45', linewidth=0.9,
                           linestyle=':', zorder=2))
    tag = ax.text(zx0, zy1, '(b)', ha='left', va='bottom', fontsize=9,
                  color='0.30', bbox=dict(fc='white', ec='none', pad=1), zorder=5)
    fig.canvas.draw()
    reserved = [tag.get_window_extent(fig.canvas.get_renderer()).padded(3).extents]

    main_records, detail_records = [], []
    for corpus, points, words in (('online', pop_2d, path_pop),
                                  ('debate', inst_2d, path_inst)):
        inside = _inside_box(points, zoom_box)
        for i, (point, word) in enumerate(zip(points, words)):
            record = dict(point=point, word=word, corpus=corpus,
                          endpoint=i in (0, len(words) - 1))
            (detail_records if inside[i] else main_records).append(record)
    try:
        main_labels = _place_labels(ax, main_records, all_points, segments, reserved)
        detail_labels = _place_labels(detail, detail_records, all_points, segments)
        counts = [_validate_labels(fig, panel, labels)
                  for panel, labels in ((ax, main_labels), (detail, detail_labels))]
        assert sum(counts) == len(path_pop) + len(path_inst)
        filename = os.path.join(IMG_DIR, f'{word1}_{word2}_discursive_paths_comparison.{OUTPUT_FORMAT}')
        fig.savefig(filename, format=OUTPUT_FORMAT, metadata={
            'Title': f'Caminhos discursivos: {word1} → {word2}',
            'Creator': os.path.basename(__file__), 'Producer': 'Matplotlib',
            'CreationDate': None, 'ModDate': None,
        })
    finally:
        plt.close(fig)

    if DELETE_STALE_PNG_FILES:
        stale_png = os.path.splitext(filename)[0] + '.png'
        if os.path.exists(stale_png):
            os.remove(stale_png)
    print(f'  Figura salva: {filename}')
    print(f'  Verificação: {sum(counts)} rótulos, sem sobreposição ou corte.')
    return filename


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
        print(f'  Online:  {" → ".join(path_pop)}')
        print(f'  Debate:  {" → ".join(path_inst)}')
        plot_comparison(model_comentarios, model_debates, path_pop, path_inst, word1, word2)

    print(f'Manifesto de caminhos: {PATH_MANIFEST}')
    print('Saídas em PDF gravadas na pasta img/.')


if __name__ == '__main__':
    main()
