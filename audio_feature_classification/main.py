import sys
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, mean_absolute_error

sys.path.append("../")
from accuracy import Accuracy, AccuracyRegression
from voice_feature_extraction import OpenSMILE
from pytorch.common.datasets_parsers.av_parser import AVDBParser


def get_data(dataset_root, file_list, max_num_clips=0):
    parser = AVDBParser(dataset_root, file_list, max_num_clips=max_num_clips)
    data = parser.get_data()
    print("clips:", len(data), "frames:", parser.get_dataset_size())
    return data


def calc_features(data, opensmile_root_dir, opensmile_config_path, task_type="classification"):
    vfe, features, targets, valid_clips = OpenSMILE(opensmile_root_dir, opensmile_config_path), [], [], []
    for clip in tqdm(data, desc="Calculating features", unit="files"):
        try:
            feat = np.asarray(vfe.process(clip.wav_rel_path), dtype=np.float32).reshape(-1)
            feat[~np.isfinite(feat)] = np.nan
            target = clip.labels if task_type == "classification" else [clip.valence, clip.arousal]
            features.append(feat)
            targets.append(target)
            valid_clips.append(clip)
        except Exception as e:
            print(f"Error processing {clip.wav_rel_path}: {e}")
    print("Processed clips:", len(features))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), valid_clips


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim=0, classifier="svm"):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    y_train, y_test = np.asarray(y_train).reshape(-1).astype(int), np.asarray(y_test).reshape(-1).astype(int)
    X_train[~np.isfinite(X_train)], X_test[~np.isfinite(X_test)] = np.nan, np.nan
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if pca_dim > 0:
        steps.append(("pca", PCA(n_components=min(pca_dim, X_train.shape[0] - 1, X_train.shape[1]), random_state=42)))
    models = {
        "svm": SVC(kernel="rbf", C=3, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    if classifier not in models:
        raise ValueError(f"Unknown classifier: {classifier}")
    model = Pipeline(steps + [("classifier", models[classifier])])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {classifier} ===")
    #print(classification_report(y_test, pred, zero_division=0))
    accuracy_fn.by_clips(pred)
    return model, pred


def regression(X_train, X_test, y_train, y_test, accuracy_fn, model_type="extra_trees"):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    y_train, y_test = np.asarray(y_train, np.float32).reshape(-1, 2), np.asarray(y_test, np.float32).reshape(-1, 2)
    X_train[~np.isfinite(X_train)], X_test[~np.isfinite(X_test)] = np.nan, np.nan
    models = {
        "extra_trees": ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "svr": make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(kernel="rbf", C=3, epsilon=0.1))),
    }
    if model_type not in models:
        raise ValueError(f"Unknown regressor: {model_type}")
    model = make_pipeline(SimpleImputer(strategy="median"), models[model_type])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Valence MAE:", mean_absolute_error(y_test[:, 0], pred[:, 0]))
    print("Arousal MAE:", mean_absolute_error(y_test[:, 1], pred[:, 1]))
    accuracy_fn.by_clips(y_test, pred)
    return model, pred


if __name__ == "__main__":
    experiment_name, max_num_clips, use_dump = "exp_1", 0, False
    task_type = "classification"  # "classification", "regression"

    opensmile_root_dir = Path("/content/opensmile-2.3.0")
    opensmile_config_path = opensmile_root_dir / "config/IS13_ComParE.conf"

    base_dir = Path("/content/INT/data/OMGEmotionChallenge")
    train_root, test_root = base_dir / "omg_TrainVideos/frames", base_dir / "omg_ValidVideos/frames"
    train_list, test_list = base_dir / "omg_TrainVideos/train_data_with_landmarks.txt", base_dir / "omg_ValidVideos/valid_data_with_landmarks.txt"

    dump_path = Path(f"{experiment_name}_{task_type}.pickle")
    if not use_dump:
        train_feat, train_y, train_data = calc_features(get_data(train_root, train_list, max_num_clips), opensmile_root_dir, opensmile_config_path, task_type)
        test_feat, test_y, test_data = calc_features(get_data(test_root, test_list, max_num_clips), opensmile_root_dir, opensmile_config_path, task_type)
        accuracy_fn = Accuracy(test_data, experiment_name) if task_type == "classification" else AccuracyRegression(test_data)
        with open(dump_path, "wb") as f:
            pickle.dump([train_feat, train_y, test_feat, test_y, accuracy_fn], f)
    else:
        with open(dump_path, "rb") as f:
            train_feat, train_y, test_feat, test_y, accuracy_fn = pickle.load(f)

    if task_type == "classification":
        classification(train_feat, test_feat, train_y, test_y, accuracy_fn, classifier="svm")
        #classification(train_feat, test_feat, train_y, test_y, accuracy_fn, classifier="extra_trees")
        #classification(train_feat, test_feat, train_y, test_y, accuracy_fn, classifier="random_forest")
    else:
        regression(train_feat, test_feat, train_y, test_y, accuracy_fn, model_type="extra_trees")
        regression(train_feat, test_feat, train_y, test_y, accuracy_fn, model_type="svr")