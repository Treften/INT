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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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



def geometry_features(p):
    def d(i, j):
        return np.linalg.norm(p[i] - p[j])

    eps = 1e-6
    eye = d(36, 45) + eps
    mouth = d(48, 54) + eps
    face = d(0, 16) + eps

    return np.asarray([
        d(62, 66) / mouth,
        d(63, 65) / mouth,
        d(51, 57) / mouth,
        d(37, 41) / eye,
        d(38, 40) / eye,
        d(43, 47) / eye,
        d(44, 46) / eye,
        d(21, 22) / eye,
        d(19, 24) / eye,
        d(27, 30) / eye,
        mouth / face,
        d(31, 35) / face,
        abs(d(37, 41) - d(44, 46)) / eye,
        abs(d(38, 40) - d(43, 47)) / eye,
        abs(d(36, 39) - d(42, 45)) / eye,
        abs(d(48, 51) - d(51, 54)) / mouth,
    ], dtype=np.float32)


def calc_features(data):
    feat, targets = [], []
    clip_feat, clip_va_targets = [], []

    pairs_68 = [
        (36, 39), (42, 45), (37, 41), (38, 40),
        (43, 47), (44, 46), (21, 22), (19, 24),
        (27, 30), (31, 35), (48, 54), (51, 57),
        (62, 66), (63, 65), (48, 60), (54, 64),
        (17, 26), (36, 45), (39, 42),
    ]

    for clip in tqdm(data, desc="calc video features", unit="files"):
        rm_list = []
        clip_features = []
        prev_points = None

        for sample in clip.data_samples:
            try:
                points = np.asarray(sample.landmarks, dtype=np.float32)

                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or len(points) < 68
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

                p = (points[:68] - center) / scale

                distances = np.asarray(
                    [np.linalg.norm(p[a] - p[b]) for a, b in pairs_68],
                    dtype=np.float32,
                )

                hull = cv2.convexHull(p.astype(np.float32))
                area = cv2.contourArea(hull)
                perimeter = cv2.arcLength(hull, True)

                base = np.concatenate([
                    p.flatten(),
                    distances,
                    np.asarray([
                        area,
                        perimeter,
                        area / (perimeter ** 2 + 1e-8),
                    ], dtype=np.float32),
                    geometry_features(p),
                ])

                delta = (
                    np.zeros_like(p)
                    if prev_points is None
                    else p - prev_points
                )

                temporal = np.concatenate([
                    delta.flatten(),
                    np.abs(delta).mean(axis=0),
                ])

                feature = np.concatenate([base, temporal])

                if not np.isfinite(feature).all():
                    rm_list.append(sample)
                    continue

                feat.append(feature)
                targets.append(int(sample.labels))
                clip_features.append(feature)
                prev_points = p.copy()

            except (ValueError, TypeError, IndexError):
                rm_list.append(sample)

        if clip_features:
            clip_feat.append(np.mean(clip_features, axis=0))
            clip_va_targets.append([clip.valence, clip.arousal])

        for sample in rm_list:
            clip.data_samples.remove(sample)

    if not feat:
        raise ValueError("Не удалось извлечь ни одного признакового вектора.")

    print("feat count:", len(feat))
    print("feature dimension:", len(feat[0]))
    print("targets count:", len(targets))
    print("clip feature count:", len(clip_feat))

    return (
        np.asarray(feat, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(clip_feat, dtype=np.float32),
        np.asarray(clip_va_targets, dtype=np.float32),
    )



def classification(
    X_train, X_test, y_train, y_test,
    accuracy_fn=None, pca_dim=0,
):
    models = {
        "SVM": make_pipeline(
            StandardScaler(),
            *([PCA(n_components=min(
                pca_dim, X_train.shape[1], X_train.shape[0] - 1
            ))] if pca_dim > 0 else []),
            SVC(C=3.0, kernel="rbf", class_weight="balanced"),
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=3000, class_weight="balanced"
            ),
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "pred": pred,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(
                y_test, pred, average="macro", zero_division=0
            ),
        }

        print(
            f"\n{name}: "
            f"accuracy={results[name]['accuracy']:.4f}, "
            f"macro-F1={results[name]['macro_f1']:.4f}"
        )

    print("\nСравнение моделей:")
    for name, result in sorted(
        results.items(),
        key=lambda item: item[1]["macro_f1"],
        reverse=True,
    ):
        print(
            f"{name:22s} "
            f"accuracy={result['accuracy']:.4f} "
            f"macro-F1={result['macro_f1']:.4f}"
        )

    # Не выбираем лучшую модель по тестовой выборке автоматически.
    # Для детального отчёта можно указать модель отдельно.
    name = "SVM"
    model = results[name]["model"]
    y_pred = results[name]["pred"]

    print(f"\nClassification report: {name}")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    if accuracy_fn is not None:
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
    train_dataset_root = (base_dir / "omg_TrainVideos/frames")
    train_file_list = (
        base_dir/ "omg_TrainVideos/train_data_with_landmarks.txt")
    test_dataset_root = (
        base_dir / "omg_ValidVideos/frames")
    test_file_list = (
        base_dir/ "omg_ValidVideos/valid_data_with_landmarks.txt")


    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root, train_file_list, max_num_clips=1000)
        test_data = get_data(test_dataset_root, test_file_list, max_num_clips=200)

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
    classification( train_feat, test_feat, train_targets, test_targets, accuracy_fn=accuracy_fn, pca_dim=0)
    regression(train_clip_feat, test_clip_feat,train_va_targets, test_va_targets,accuracy_fn=accuracy_va)
