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
    """Plot the full discursive paths plus an automatic picture-in-picture zoom.

    The saved filename is intentionally identical to the original script:
    img/{word1}_{word2}_discursive_paths_comparison.png
    """
    from matplotlib.patches import Rectangle

    os.makedirs('img', exist_ok=True)

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
        # Select the dense central mass, leaving extreme endpoints/outliers on the macro plot.
        # This is a crop in the same PCA coordinate system, not a recomputation or distortion.
        if len(points) <= 8:
            core = points
        else:
            center = np.median(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            n_core = int(np.ceil(len(points) * 0.72))
            n_core = max(8, min(len(points), n_core))
            core = points[np.argsort(distances)[:n_core]]

        xmin, xmax = _finite_range(core[:, 0])
        ymin, ymax = _finite_range(core[:, 1])

        full_xmin, full_xmax, full_ymin, full_ymax = _limits(points, pad_fraction=0.02)
        full_width = full_xmax - full_xmin
        full_height = full_ymax - full_ymin

        width = xmax - xmin
        height = ymax - ymin
        xpad = max(width * 0.28, full_width * 0.035, 0.04)
        ypad = max(height * 0.35, full_height * 0.035, 0.04)

        xmin -= xpad
        xmax += xpad
        ymin -= ypad
        ymax += ypad

        # Avoid an inset that is almost as large as the full plot; it stops being a zoom.
        max_zoom_width = full_width * 0.62
        max_zoom_height = full_height * 0.62
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
            (0.545, 0.525, 0.425, 0.405),  # upper right
            (0.055, 0.525, 0.425, 0.405),  # upper left
            (0.545, 0.075, 0.425, 0.405),  # lower right
            (0.055, 0.075, 0.425, 0.405),  # lower left
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
            # Prefer spots with fewer points underneath; slightly prefer upper positions.
            score = int(np.sum(covered)) - (0.25 if by > 0.5 else 0.0)
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
        if arrows:
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
                # In zoomed insets, paths often continue to far-away endpoints.
                # Skip those arrowheads so Matplotlib does not draw giant off-panel arrows.
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
            base_offset = 0.105 if len(labels) <= 18 else 0.135
            preferred = anchors_ax + unit * base_offset
        else:
            preferred = anchors_ax + unit * 0.045

        positions = preferred.copy()

        widths = np.array([0.026 + 0.0062 * min(len(str(s)), 18) for s in labels])
        heights = np.full(len(labels), 0.035 if fontsize <= 8 else 0.043)
        if mode == 'inset':
            heights *= 1.05

        # Deterministic mini force layout in axis coordinates. It does not move data points;
        # it only moves the text callouts and draws leader lines back to the true points.
        for _ in range(260):
            movement = (preferred - positions) * 0.015
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

    fig, ax = plt.subplots(figsize=(16, 9.5))
    fig.patch.set_facecolor('white')

    full_xlim = _limits(all_points, pad_fraction=0.14)[:2]
    full_ylim_tuple = _limits(all_points, pad_fraction=0.14)[2:]
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
        bbox_to_anchor=(0.5, 1.03),
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
        spine.set_linewidth(1.0)
        spine.set_edgecolor('0.20')

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)

    _plot_path(axins, pop_2d, '-', 'o', label=None, linewidth=1.65, alpha=0.90, arrows=True)
    _plot_path(axins, inst_2d, '--', 's', label=None, linewidth=1.65, alpha=0.90, arrows=True)
    axins.set_aspect('equal', adjustable='box')
    axins.grid(True, alpha=0.22, linewidth=0.6)
    axins.tick_params(labelsize=7, length=2.5, pad=1)
    axins.set_title('zoom da zona densa', fontsize=9.5, pad=4)

    inset_pop_idx = [i for i, inside in enumerate(pop_in_zoom) if inside]
    inset_inst_idx = [i for i, inside in enumerate(inst_in_zoom) if inside]

    _annotate_words(axins, pop_2d[inset_pop_idx], [path_pop[i] for i in inset_pop_idx], fontsize=7.4, mode='inset')
    _annotate_words(axins, inst_2d[inset_inst_idx], [path_inst[i] for i in inset_inst_idx], fontsize=7.4, mode='inset')

    filename = f'img/{word1}_{word2}_discursive_paths_comparison.png'
    fig.savefig(filename, dpi=350, bbox_inches='tight')
    plt.close(fig)

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
