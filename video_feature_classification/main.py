import sys

sys.path.append("../")

import os
import pickle
import random

from pathlib import Path


from accuracy import Accuracy, AccuracyRegression
import cv2
import numpy as np
from tqdm import tqdm
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

def get_data(dataset_root, file_list, max_num_clips=0, max_num_samples=50):
    dataset_parser = AVDBParser(
        dataset_root,
        os.path.join(dataset_root, file_list),
        max_num_clips=max_num_clips,
        max_num_samples=max_num_samples,
        ungroup=False,
        load_image=True,
    )
    data = dataset_parser.get_data()
    print("clips count:", len(data))
    print("frames count:", dataset_parser.get_dataset_size())
    return data



def calc_features(data):
    feat, targets, va_targets = [], [], []
    clip_feat, clip_va_targets = [], []

    # Индексы для стандартной разметки лица из 68 точек.
    pairs_68 = [
        (36, 39), (42, 45),  # ширина глаз
        (37, 41), (38, 40),  # раскрытие левого глаза
        (43, 47), (44, 46),  # раскрытие правого глаза
        (21, 22), (19, 24),  # расстояния между бровями
        (27, 30), (31, 35),  # нос
        (48, 54), (51, 57),  # ширина и раскрытие рта
        (62, 66), (63, 65),  # внутреннее раскрытие рта
        (48, 60), (54, 64),
        (17, 26), (21, 22),
        (36, 45), (39, 42),
    ]

    for clip in tqdm(data, desc="calc video features", unit="files"):
        rm_list = []
        clip_features = []
        for sample in clip.data_samples:
            try:
                points = np.asarray(sample.landmarks, dtype=np.float32)

                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or len(points) < 3
                    or not np.isfinite(points).all()
                    or sample.labels is None
                ):
                    rm_list.append(sample)
                    continue

                
                left_eye = points[36:42].mean(axis=0)
                right_eye = points[42:48].mean(axis=0)
                center = (left_eye + right_eye) / 2
                scale = np.linalg.norm(right_eye - left_eye)


                if not np.isfinite(scale) or scale < 1e-6:
                    rm_list.append(sample)
                    continue

                p = (points - center) / scale

                features = [p.flatten()]

                distances = np.array([np.linalg.norm(p[a] - p[b])for a, b in pairs_68], dtype=np.float32)
                features.append(distances)

                hull = cv2.convexHull(p.astype(np.float32))
                area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(hull, True)

                hull_features = np.array([area,perimeter,area / (perimeter ** 2 + 1e-8),], dtype=np.float32)

                features.append(hull_features)
                feature = np.concatenate(features)
                feat.append(feature)
                targets.append(int(sample.labels))
                clip_features.append(feature)
                clip_features.append(feature)
                if not np.isfinite(feature).all():
                    rm_list.append(sample)
                    continue
                
            except (ValueError, TypeError, IndexError):
                rm_list.append(sample)
        if clip_features:
            clip_feat.append(np.mean(clip_features, axis=0))
            clip_va_targets.append([clip.valence, clip.arousal])

        for sample in rm_list:
            clip.data_samples.remove(sample)

    if not feat:
        raise ValueError("Не удалось извлечь ни одного признакового вектора.")

    lengths = {len(x) for x in feat}
    if len(lengths) != 1:
        raise ValueError(
            "Разное количество landmarks в кадрах: "
            "признаки имеют разную размерность."
        )

    print("feat count:", len(feat))
    print("feature dimension:", len(feat[0]))
    print("targets count:", len(targets))

    return (
        np.asarray(feat, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(clip_feat, dtype=np.float32),
        np.asarray(clip_va_targets, dtype=np.float32),
    )



def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim: int = 0):
    steps = [StandardScaler()]
    if pca_dim > 0:
        steps.append(PCA(n_components=min(pca_dim, X_train.shape[1], X_train.shape[0] - 1)))
    steps.append(SVC(C=3.0, kernel="rbf", class_weight="balanced"))

    model = make_pipeline(*steps)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Test accuracy:", np.mean(y_pred == y_test))
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)
    return model, y_pred

def regression(X_train, X_test, y_train, y_test, accuracy_fn):
    model = make_pipeline(
        StandardScaler(),
        ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    for i, name in enumerate(["Valence", "Arousal"]):
        print(
            f"{name}: MAE={mean_absolute_error(y_test[:, i], y_pred[:, i]):.4f}, "
            f"RMSE={np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])):.4f}, "
            f"R2={r2_score(y_test[:, i], y_pred[:, i]):.4f}"
        )

    accuracy_fn.by_clips(y_test, y_pred)
    return model, y_pred

if __name__ == "__main__":
    experiment_name = "exp_1"
    max_num_clips = 0  # загружайте только часть данных для отладки кода
    use_dump = False  # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = Path("/content/INT/data/OMGEmotionChallenge")
    train_dataset_root = (
        base_dir / "omg_TrainVideos/frames"
        )
    train_file_list = (
        base_dir
            / "omg_TrainVideos/train_data_with_landmarks.txt"
        )
    test_dataset_root = (
        base_dir / "omg_ValidVideos/frames"
    )
    test_file_list = (
        base_dir
        / "omg_ValidVideos/valid_data_with_landmarks.txt"
    )


    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=1200)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=300)

        # get features
        train_feat, train_targets, train_clip_feat, train_va_targets = calc_features(train_data)
        test_feat, test_targets, test_clip_feat, test_va_targets = calc_features(test_data)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)
        accuracy_va = AccuracyRegression(test_data, threshold=0.1)
        # with open(experiment_name + '.pickle', 'wb') as f:
        #    pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + ".pickle", "rb") as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(
                f
            )

    # run classifiers
    classification( train_feat,test_feat,train_targets,test_targets,accuracy_fn=accuracy_fn,pca_dim=0,)
    regression(train_clip_feat, test_clip_feat,train_va_targets, test_va_targets,accuracy_fn=accuracy_va)
