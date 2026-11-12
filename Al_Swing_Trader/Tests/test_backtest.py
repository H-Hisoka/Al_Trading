
"""
Kiểm thử hệ thống Backtest.

Modules:
- backtest/metrics.py
- backtest/simulator.py
- backtest/walk_forward.py

Mục tiêu:
1. Kiểm tra mô phỏng dự đoán trên dữ liệu lịch sử.
2. Kiểm tra các chỉ số đánh giá mô hình.
3. Kiểm tra chia dữ liệu Walk-forward.
4. Đảm bảo không shuffle dữ liệu thời gian.
5. Đảm bảo không có lookahead bias.
6. Đảm bảo train/test không bị chồng lấn.

Chạy:
    python -m pytest tests/test_backtest.py -v
"""

import numpy as np
import pandas as pd
import pytest

try:
    from backtest.simulator import (
        run_backtest,
        simulate,
        save_backtest_results,
    )

    from backtest.metrics import (
        calculate_classification_metrics,
        calculate_probability_metrics,
        calculate_calibration_report,
        calculate_swing_metrics,
        evaluate,
        save_metrics_report,
    )

    from backtest.walk_forward import (
        WalkForwardConfig,
        generate_splits,
        run_walk_forward,
        save_walk_forward_results,
    )

except ImportError:
    from ..backtest.simulator import (
        run_backtest,
        simulate,
        save_backtest_results,
    )

    from ..backtest.metrics import (
        calculate_classification_metrics,
        calculate_probability_metrics,
        calculate_calibration_report,
        calculate_swing_metrics,
        evaluate,
        save_metrics_report,
    )

    from ..backtest.walk_forward import (
        WalkForwardConfig,
        generate_splits,
        run_walk_forward,
        save_walk_forward_results,
    )


# ============================================================
# CONSTANTS
# ============================================================

CLASSES = [
    "TOP_SOON",
    "BOTTOM_SOON",
    "NONE",
]


# ============================================================
# FIXTURES
# ============================================================

def make_features(n=300, seed=42):
    """Tạo feature giả lập phục vụ unit test."""
    rng = np.random.default_rng(seed)

    index = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "atr": rng.uniform(0.1, 1.0, n),
            "rsi": rng.uniform(20, 80, n),
            "ema_distance": rng.normal(0, 1, n),
            "candle_body": rng.uniform(0, 1, n),
            "volume_ratio": rng.uniform(0.5, 2.0, n),
        },
        index=index,
    )


def make_labels(n=300, seed=42):
    rng = np.random.default_rng(seed)

    labels = rng.choice(
        CLASSES,
        size=n,
        p=[0.25, 0.25, 0.50],
    )

    index = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="5min",
    )

    return pd.Series(
        labels,
        index=index,
        name="future_label",
    )


def make_backtest_results(n=100, seed=42):
    """Tạo kết quả dự đoán giả lập để kiểm tra metrics."""
    rng = np.random.default_rng(seed)

    index = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="5min",
    )

    actual = rng.choice(
        CLASSES,
        size=n,
        p=[0.25, 0.25, 0.50],
    )

    probabilities = rng.dirichlet(
        [2, 2, 4],
        size=n,
    )

    predicted = np.array(CLASSES)[
        probabilities.argmax(axis=1)
    ]

    return pd.DataFrame(
        {
            "actual_label": actual,
            "p_top_soon": probabilities[:, 0],
            "p_bottom_soon": probabilities[:, 1],
            "p_none": probabilities[:, 2],
            "predicted_class": predicted,
            "signal_status": predicted,
            "prediction_time": index,
            "error": None,
        },
        index=pd.Index(
            index,
            name="timestamp",
        ),
    )


@pytest.fixture
def features():
    return make_features()


@pytest.fixture
def labels():
    return make_labels()


@pytest.fixture
def results():
    return make_backtest_results()


# ============================================================
# FAKE PREDICTOR
# ============================================================

class FakePredictor:
    """
    Predictor giả lập.

    Không train model thật và không kết nối MT5.
    """

    def __init__(self):
        self.predict_count = 0
        self.seen_indexes = []

    def predict_proba(self, X):
        self.predict_count += 1
        self.seen_indexes.append(X.index.copy())

        n = len(X)

        probabilities = np.tile(
            [0.20, 0.30, 0.50],
            (n, 1),
        )

        return pd.DataFrame(
            probabilities,
            index=X.index,
            columns=CLASSES,
        )


# ============================================================
# METRICS: CLASSIFICATION
# ============================================================

def test_classification_metrics_return_dict(results):
    report = calculate_classification_metrics(
        results.copy()
    )

    assert isinstance(report, dict)


def test_classification_metrics_have_basic_scores(
    results,
):
    report = calculate_classification_metrics(
        results.copy()
    )

    expected = {
        "accuracy",
        "precision",
        "recall",
        "f1",
    }

    # Cho phép tên metric có hậu tố macro/weighted.
    keys = {
        str(key).lower()
        for key in report
    }

    for metric in expected:
        assert any(
            metric in key
            for key in keys
        ), f"Thiếu chỉ số {metric}"


def test_classification_metrics_are_finite(results):
    report = calculate_classification_metrics(
        results.copy()
    )

    for value in report.values():
        if isinstance(value, (int, float, np.number)):
            assert np.isfinite(value)


def test_classification_metrics_reject_empty_data():
    empty = pd.DataFrame(
        columns=[
            "actual_label",
            "predicted_class",
        ]
    )

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        calculate_classification_metrics(empty)


# ============================================================
# METRICS: PROBABILITY
# ============================================================

def test_probability_metrics_return_dict(results):
    report = calculate_probability_metrics(
        results.copy()
    )

    assert isinstance(report, dict)


def test_probability_metrics_are_finite(results):
    report = calculate_probability_metrics(
        results.copy()
    )

    for value in report.values():
        if isinstance(value, (int, float, np.number)):
            assert np.isfinite(value)


def test_probability_columns_are_valid(results):
    probability_columns = [
        "p_top_soon",
        "p_bottom_soon",
        "p_none",
    ]

    for col in probability_columns:
        assert col in results.columns

        assert (
            results[col].between(0, 1)
        ).all()


def test_probabilities_sum_to_one(results):
    total = (
        results["p_top_soon"]
        + results["p_bottom_soon"]
        + results["p_none"]
    )

    assert np.allclose(
        total.to_numpy(),
        1.0,
        atol=1e-6,
    )


def test_probability_metrics_reject_invalid_probabilities(
    results,
):
    invalid = results.copy()

    invalid.loc[
        invalid.index[0],
        "p_top_soon",
    ] = 2.0

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        calculate_probability_metrics(invalid)


# ============================================================
# METRICS: CALIBRATION
# ============================================================

def test_calibration_report_returns_dataframe(results):
    report = calculate_calibration_report(
        results.copy(),
        n_bins=10,
    )

    assert isinstance(report, pd.DataFrame)


def test_calibration_report_has_valid_bin_count(
    results,
):
    report = calculate_calibration_report(
        results.copy(),
        n_bins=10,
    )

    assert len(report) <= 10


def test_calibration_report_rejects_invalid_bins(
    results,
):
    with pytest.raises(
        (ValueError, AssertionError):
    ):
        calculate_calibration_report(
            results.copy(),
            n_bins=0,
        )


# ============================================================
# METRICS: SWING
# ============================================================

def test_swing_metrics_return_dict(results):
    report = calculate_swing_metrics(
        results.copy()
    )

    assert isinstance(report, dict)


def test_swing_metrics_are_finite(results):
    report = calculate_swing_metrics(
        results.copy()
    )

    for value in report.values():
        if isinstance(value, (int, float, np.number)):
            assert np.isfinite(value)


# ============================================================
# METRICS: EVALUATE
# ============================================================

def test_evaluate_returns_dict(results):
    report = evaluate(
        results.copy()
    )

    assert isinstance(report, dict)


def test_evaluate_contains_all_sections(results):
    report = evaluate(
        results.copy()
    )

    expected_sections = {
        "sample_count",
        "classification",
        "probability",
        "swing",
        "calibration",
    }

    assert expected_sections.issubset(
        report.keys()
    )


def test_evaluate_sample_count_is_correct(results):
    report = evaluate(
        results.copy()
    )

    assert report["sample_count"] == len(results)


# ============================================================
# SIMULATOR
# ============================================================

def test_simulator_returns_dataframe(features, labels):
    predictor = FakePredictor()

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        strict=True,
    )

    assert isinstance(result, pd.DataFrame)


def test_simulator_contains_required_columns(
    features,
    labels,
):
    predictor = FakePredictor()

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        strict=True,
    )

    required = {
        "actual_label",
        "p_top_soon",
        "p_bottom_soon",
        "p_none",
        "predicted_class",
        "signal_status",
        "prediction_time",
        "error",
    }

    assert required.issubset(
        result.columns
    )


def test_simulator_preserves_prediction_order(
    features,
    labels,
):
    predictor = FakePredictor()

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        strict=True,
    )

    assert result.index.is_monotonic_increasing


def test_simulator_does_not_duplicate_timestamps(
    features,
    labels,
):
    predictor = FakePredictor()

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        strict=True,
    )

    assert result.index.is_unique


def test_simulator_predictions_have_valid_classes(
    features,
    labels,
):
    predictor = FakePredictor()

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        strict=True,
    )

    assert set(
        result["predicted_class"].dropna().unique()
    ).issubset(set(CLASSES))


def test_simulator_probabilities_are_between_zero_and_one(
    features,
    labels,
):
    predictor = FakePredictor()

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        strict=True,
    )

    for col in (
        "p_top_soon",
        "p_bottom_soon",
        "p_none",
    ):
        assert result[col].dropna().between(0, 1).all()


def test_simulator_does_not_modify_features(
    features,
    labels,
):
    original = features.copy(deep=True)

    predictor = FakePredictor()

    run_backtest(
        features,
        labels=labels,
        predictor=predictor,
        strict=True,
    )

    pd.testing.assert_frame_equal(
        features,
        original,
    )


def test_simulator_does_not_modify_labels(
    features,
    labels,
):
    original = labels.copy(deep=True)

    predictor = FakePredictor()

    run_backtest(
        features,
        labels=labels,
        predictor=predictor,
        strict=True,
    )

    pd.testing.assert_series_equal(
        labels,
        original,
    )


def test_simulator_respects_start_and_end_positions(
    features,
    labels,
):
    predictor = FakePredictor()

    start = 30
    end = 100

    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        start_position=start,
        end_position=end,
        strict=True,
    )

    assert len(result) <= end - start

    assert (
        result.index.min()
        >= features.index[start]
    )

    assert (
        result.index.max()
        < features.index[end]
    )


def test_simulator_rejects_misaligned_labels(features):
    wrong_labels = make_labels(
        n=len(features) - 1
    )

    predictor = FakePredictor()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        run_backtest(
            features.copy(),
            labels=wrong_labels,
            predictor=predictor,
            strict=True,
        )


def test_simulator_rejects_unsorted_features(
    features,
    labels,
):
    unsorted = features.iloc[::-1].copy()

    predictor = FakePredictor()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        run_backtest(
            unsorted,
            labels=labels,
            predictor=predictor,
            strict=True,
        )


def test_simulator_alias_matches_main_function(
    features,
    labels,
):
    first = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=FakePredictor(),
        strict=True,
    )

    second = simulate(
        features.copy(),
        labels=labels.copy(),
        predictor=FakePredictor(),
        strict=True,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


# ============================================================
# SIMULATOR: NO LOOKAHEAD
# ============================================================

def test_simulator_does_not_predict_before_start_position(
    features,
    labels,
):
    predictor = FakePredictor()

    start = 50

    run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=predictor,
        start_position=start,
        strict=True,
    )

    for seen_index in predictor.seen_indexes:
        assert (
            seen_index.min()
            >= features.index[start]
        )


def test_simulator_future_changes_do_not_change_past_predictions(
    features,
    labels,
):
    """
    Thay đổi dữ liệu sau cutoff không được thay đổi
    kết quả dự đoán trước hoặc tại cutoff.

    Test này kiểm tra predictor giả lập và luồng
    simulator. Muốn kiểm tra model thực tế phải chạy
    thêm với artifact đã train.
    """
    cutoff_position = 150
    cutoff = features.index[cutoff_position]

    first = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=FakePredictor(),
        strict=True,
    )

    changed = features.copy()

    changed.loc[
        changed.index[cutoff_position + 1:],
        "atr",
    ] += 1000

    second = run_backtest(
        changed,
        labels=labels.copy(),
        predictor=FakePredictor(),
        strict=True,
    )

    pd.testing.assert_frame_equal(
        first.loc[:cutoff],
        second.loc[:cutoff],
        check_dtype=False,
    )


# ============================================================
# WALK-FORWARD: SPLITS
# ============================================================

def make_walk_forward_config():
    return WalkForwardConfig(
        n_splits=3,
        test_size=20,
        min_train_size=50,
        train_window=None,
        purge_size=5,
        embargo_size=0,
        step_size=20,
        window_type="expanding",
        start_position=0,
    )


def test_generate_splits_returns_list(features):
    config = make_walk_forward_config()

    folds = generate_splits(
        features,
        config,
    )

    assert isinstance(folds, list)
    assert len(folds) > 0


def test_walk_forward_folds_are_chronological(features):
    config = make_walk_forward_config()

    folds = generate_splits(
        features,
        config,
    )

    for fold in folds:
        assert (
            fold.train_index.max()
            < fold.test_index.min()
        )


def test_walk_forward_train_test_do_not_overlap(
    features,
):
    config = make_walk_forward_config()

    folds = generate_splits(
        features,
        config,
    )

    for fold in folds:
        train_set = set(fold.train_index)
        test_set = set(fold.test_index)

        assert train_set.isdisjoint(test_set)


def test_walk_forward_has_purge_gap(features):
    purge_size = 5

    config = make_walk_forward_config()

    folds = generate_splits(
        features,
        config,
    )

    for fold in folds:
        train_positions = features.index.get_indexer(
            fold.train_index
        )

        test_positions = features.index.get_indexer(
            fold.test_index
        )

        gap = (
            test_positions.min()
            - train_positions.max()
            - 1
        )

        assert gap >= purge_size


def test_walk_forward_uses_expanding_training_window(
    features,
):
    config = make_walk_forward_config()

    folds = generate_splits(
        features,
        config,
    )

    if len(folds) < 2:
        pytest.skip("Cần ít nhất hai fold.")

    first_train_size = len(
        folds[0].train_index
    )

    second_train_size = len(
        folds[1].train_index
    )

    assert second_train_size >= first_train_size


def test_walk_forward_test_sets_are_not_overlapping(
    features,
):
    config = make_walk_forward_config()

    folds = generate_splits(
        features,
        config,
    )

    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            first = set(folds[i].test_index)
            second = set(folds[j].test_index)

            assert first.isdisjoint(second)


def test_walk_forward_rejects_unsorted_data(features):
    unsorted = features.iloc[::-1].copy()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        generate_splits(
            unsorted,
            make_walk_forward_config(),
        )


def test_walk_forward_rejects_duplicate_index(features):
    duplicated = pd.concat(
        [
            features,
            features.iloc[[10]],
        ]
    ).sort_index()

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        generate_splits(
            duplicated,
            make_walk_forward_config(),
        )


# ============================================================
# WALK-FORWARD: TRAIN / PREDICT
# ============================================================

def test_run_walk_forward_returns_dict(features, labels):
    config = make_walk_forward_config()

    def train_fn(X_train, y_train, fold):
        return {
            "train_size": len(X_train),
            "fold_id": fold.fold_id,
        }

    def predict_fn(model, X_test, fold):
        n = len(X_test)

        return pd.DataFrame(
            {
                "predicted_class": ["NONE"] * n,
                "p_top_soon": [0.1] * n,
                "p_bottom_soon": [0.1] * n,
                "p_none": [0.8] * n,
            },
            index=X_test.index,
        )

    result = run_walk_forward(
        features,
        labels,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=config,
    )

    assert isinstance(result, dict)


def test_run_walk_forward_contains_expected_keys(
    features,
    labels,
):
    config = make_walk_forward_config()

    def train_fn(X_train, y_train, fold):
        return object()

    def predict_fn(model, X_test, fold):
        return pd.DataFrame(
            {
                "predicted_class": ["NONE"] * len(X_test),
            },
            index=X_test.index,
        )

    result = run_walk_forward(
        features,
        labels,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=config,
    )

    expected = {
        "folds",
        "predictions",
        "config",
        "n_folds_completed",
    }

    assert expected.issubset(result.keys())


def test_run_walk_forward_trains_once_per_fold(
    features,
    labels,
):
    config = make_walk_forward_config()

    trained_folds = []

    def train_fn(X_train, y_train, fold):
        trained_folds.append(fold.fold_id)
        return object()

    def predict_fn(model, X_test, fold):
        return pd.DataFrame(
            {
                "predicted_class": ["NONE"] * len(X_test),
            },
            index=X_test.index,
        )

    result = run_walk_forward(
        features,
        labels,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=config,
    )

    assert len(trained_folds) == result[
        "n_folds_completed"
    ]


def test_run_walk_forward_predictions_are_chronological(
    features,
    labels,
):
    config = make_walk_forward_config()

    def train_fn(X_train, y_train, fold):
        return object()

    def predict_fn(model, X_test, fold):
        return pd.DataFrame(
            {
                "predicted_class": ["NONE"] * len(X_test),
            },
            index=X_test.index,
        )

    result = run_walk_forward(
        features,
        labels,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=config,
    )

    predictions = result["predictions"]

    assert predictions.index.is_monotonic_increasing


def test_run_walk_forward_never_trains_on_test_rows(
    features,
    labels,
):
    config = make_walk_forward_config()

    def train_fn(X_train, y_train, fold):
        assert (
            X_train.index.max()
            < fold.test_index.min()
        )

        assert X_train.index.intersection(
            fold.test_index
        ).empty

        return object()

    def predict_fn(model, X_test, fold):
        assert X_test.index.equals(
            fold.test_index
        )

        return pd.DataFrame(
            {
                "predicted_class": ["NONE"] * len(X_test),
            },
            index=X_test.index,
        )

    run_walk_forward(
        features,
        labels,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=config,
    )


def test_run_walk_forward_rejects_misaligned_labels(
    features,
):
    wrong_labels = make_labels(
        n=len(features) - 1
    )

    def train_fn(X_train, y_train, fold):
        return object()

    def predict_fn(model, X_test, fold):
        return pd.DataFrame(
            index=X_test.index
        )

    with pytest.raises(
        (ValueError, AssertionError)
    ):
        run_walk_forward(
            features,
            wrong_labels,
            train_fn=train_fn,
            predict_fn=predict_fn,
            config=make_walk_forward_config(),
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def test_save_backtest_results(
    features,
    labels,
    tmp_path,
):
    result = run_backtest(
        features.copy(),
        labels=labels.copy(),
        predictor=FakePredictor(),
        strict=True,
    )

    output_path = tmp_path / "backtest.csv"

    save_backtest_results(
        result,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_metrics_report(results, tmp_path):
    report = evaluate(results.copy())

    output_path = tmp_path / "metrics.json"

    save_metrics_report(
        report,
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_walk_forward_results(
    features,
    labels,
    tmp_path,
):
    config = make_walk_forward_config()

    def train_fn(X_train, y_train, fold):
        return object()

    def predict_fn(model, X_test, fold):
        return pd.DataFrame(
            {
                "predicted_class": ["NONE"] * len(X_test),
            },
            index=X_test.index,
        )

    result = run_walk_forward(
        features,
        labels,
        train_fn=train_fn,
        predict_fn=predict_fn,
        config=config,
    )

    output_path = tmp_path / "walk_forward"

    save_walk_forward_results(
        result,
        output_path,
    )

    assert output_path.exists()
    assert any(output_path.iterdir())
