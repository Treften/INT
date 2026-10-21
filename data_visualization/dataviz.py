import sys

sys.path.append("../")
import matplotlib

matplotlib.use("Agg")
import os

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from t_sne.tsne import TSNE as torchTSNE

from data_visualization.t_sne.wrapper import Wrapper
from pytorch.common.datasets_parsers.av_parser import AVDBParser


def get_data(dataset_root, file_list, max_num_clips=0):
    dataset_parser = AVDBParser(
        dataset_root,
        file_list,
        max_num_clips=max_num_clips,
        ungroup=False,
        load_image=False,
    )
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data


def dist(a, b):
    return np.linalg.norm(a - b)

def calc_features(data, draw: bool = False):
    feat, targets = [], []
    for clip in data:

        # TODO: придумайте способы вычисления признаков на основе ключевых точек
        # distance between landmarks
        for i, sample in enumerate(clip.data_samples):
            if i % 8 != 0:
                continue
            nose_dist = []
            lm_ref = sample.landmarks[30]  # point on the nose
            landmarks = np.asarray(sample.landmarks, dtype=np.float32)
            for j in range(len(sample.landmarks)):
                lm = sample.landmarks[j]
                nose_dist.append(np.sqrt((lm_ref[0] - lm[0]) ** 2 + (lm_ref[1] - lm[1]) ** 2))
            feat.append(nose_dist)
            targets.append(sample.labels)



            mouth_width = dist(landmarks[48], landmarks[54])
            mouth_height = dist(landmarks[51], landmarks[57])

            eye_left_width = dist(landmarks[36], landmarks[39])
            eye_right_width = dist(landmarks[42], landmarks[45])

            brow_left_eye = dist(landmarks[21], landmarks[39])
            brow_right_eye = dist(landmarks[22], landmarks[42])

            face_width = np.linalg.norm(landmarks[0] - landmarks[16])

            features = (landmarks - lm_ref) / (face_width + 1e-8)

            #feat.append(features.flatten())
            print("feat count:", len(feat))
            if draw:
                img = cv2.cv.LoadImage(sample.img_rel_path)
                for lm in sample.landmarks:
                    cv2.circle(img, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
                cv2.imshow(sample.text_labels, img)
                cv2.waitKey(100)

    print("feat count:", len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)



def draw(points2D, targets, sawe=True):
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        points2D[:, 0],
        points2D[:, 1],
        c=targets,
        cmap="viridis",
        s=12,
        alpha=0.7
    )

    ax.set_title("t-SNE visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Target label")

    output_path = Path("/content/INT/scatter.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved: {output_path}", flush=True)



def run_tsne(feat, targets, pca_dim=50, tsne_dim=2):
    print("run_tsne")
    if pca_dim > 0:
        print("pca")
        feat = PCA(n_components=pca_dim).fit_transform(feat)
    print("dist")
    distances2 = pairwise_distances(feat, metric="euclidean", squared=True)
    # This return a n x (n-1) prob array
    print("pij")
    pij = manifold._t_sne._joint_probabilities(distances2, 30, False)
    # Convert to n x n prob array
    pij = squareform(pij)

    i, j = np.indices(pij.shape)
    i, j = i.ravel(), j.ravel()
    pij = pij.ravel().astype("float32")
    # Remove self-indices
    idx = i != j
    i, j, pij = i[idx], j[idx], pij[idx]

    model = torchTSNE(n_points=feat.shape[0], n_dim=tsne_dim)
    print("wrapper")
    w = Wrapper(model, cuda=True, batchsize=feat.shape[0], epochs=5)
    for itr in range(15):
        w.fit(pij, i, j, itr=itr)
        # Visualize the results
    embed = model.logits.weight.cpu().data.numpy()
    draw(embed, targets)


if __name__ == "__main__":
    # dataset dir
    base_dir = Path("/content/INT/data/")
    '''if 1:
        train_dataset_root = base_dir / "Ryerson/Video"
        train_file_list = base_dir / "Ryerson/train_data_with_landmarks.txt"
    elif 0:
        train_dataset_root = base_dir / "/AFEW-VA/crop"
        train_file_list = base_dir / "AFEW-VA/crop/train_data_with_landmarks.txt"
    elif 0:'''
    train_dataset_root = (
        base_dir / "omg_TrainVideos/frames"
        )
    train_file_list = (
        base_dir
            / "omg_TrainVideos/train_data_with_landmarks.txt"
        )

    # load dataset
    data = get_data(train_dataset_root, train_file_list, max_num_clips=0)

    # get features
    feat, targets = calc_features(data, draw=False)
    print("tsne")
    # run t-SNE
    rng = np.random.default_rng(42)

    n = min(5000, len(feat))
    idx = rng.choice(len(feat), size=n, replace=False)

    feat_small = feat[idx]
    targets_small = targets[idx]

    run_tsne(feat_small, targets_small, pca_dim=0)
