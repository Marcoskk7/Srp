"""
Tests for GenerationConfig (config.py).

torch is required because GenerationConfig calls torch.device() at construction
time. Tests are skipped automatically when torch is not installed.
"""
import argparse
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping config tests")

from config import GenerationConfig  # noqa: E402  (import after torch guard)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_args(**overrides) -> argparse.Namespace:
    """Return the smallest valid Namespace that GenerationConfig accepts."""
    defaults = {}          # GenerationConfig uses getattr() fallbacks for every arg
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGenerationConfigConstruction:

    def test_builds_from_empty_namespace(self):
        """GenerationConfig must not raise when no optional args are present."""
        cfg = GenerationConfig(_minimal_args())
        assert cfg is not None

    def test_data_namespace_exists(self):
        cfg = GenerationConfig(_minimal_args())
        assert hasattr(cfg, "data"), "config.data must exist"

    def test_model_namespace_exists(self):
        cfg = GenerationConfig(_minimal_args())
        assert hasattr(cfg, "model"), "config.model must exist"

    def test_training_namespace_exists(self):
        cfg = GenerationConfig(_minimal_args())
        assert hasattr(cfg, "training"), "config.training must exist"


# ---------------------------------------------------------------------------
# config.data fields
# ---------------------------------------------------------------------------

class TestGenerationConfigData:

    def test_dataset_is_cwru(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.data.dataset == "CWRU"

    def test_task_type_is_fault(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.data.task_type == "fault"

    def test_num_classes_train_matches_source_codes(self):
        cfg = GenerationConfig(_minimal_args())
        # num_classes_train is derived by base_trainer from len(cwru_fault_source_codes)
        assert len(cfg.data.cwru_fault_source_codes) == 6

    def test_num_classes_test_matches_target_codes(self):
        cfg = GenerationConfig(_minimal_args())
        assert len(cfg.data.cwru_fault_target_codes) == 4

    def test_cwru_signal_length(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.data.cwru_signal_length == 2400

    def test_effective_signal_length_equals_cwru_signal_length(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.data.effective_signal_length == cfg.data.cwru_signal_length

    def test_pu_placeholder_fields_present(self):
        """PU fields must exist even though generation does not use them."""
        cfg = GenerationConfig(_minimal_args())
        assert hasattr(cfg.data, "pu_data_type")
        assert hasattr(cfg.data, "pu_class_num_train")
        assert hasattr(cfg.data, "pu_class_num_test")
        assert hasattr(cfg.data, "pu_signal_length")
        assert hasattr(cfg.data, "pu_fft_length")


# ---------------------------------------------------------------------------
# config.model fields
# ---------------------------------------------------------------------------

class TestGenerationConfigModel:

    def test_feature_dim(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.model.feature_dim == 64

    def test_cwru_adaptive_pool_size(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.model.cwru_adaptive_pool_size == 64

    def test_relation_dim_present(self):
        cfg = GenerationConfig(_minimal_args())
        assert hasattr(cfg.model, "relation_dim")


# ---------------------------------------------------------------------------
# config.training fields
# ---------------------------------------------------------------------------

class TestGenerationConfigTraining:

    def test_device_is_torch_device(self):
        cfg = GenerationConfig(_minimal_args())
        assert isinstance(cfg.training.device, torch.device)

    def test_learning_rate(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.training.learning_rate == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Optional args / defaults
# ---------------------------------------------------------------------------

class TestGenerationConfigDefaults:

    def test_seed_defaults_to_42(self):
        cfg = GenerationConfig(_minimal_args())  # no seed arg
        assert cfg.training.random_seed == 42

    def test_seed_override(self):
        cfg = GenerationConfig(_minimal_args(seed=7))
        assert cfg.training.random_seed == 7

    def test_force_regenerate_defaults_to_false(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.training.force_retrain is False

    def test_force_regenerate_override(self):
        cfg = GenerationConfig(_minimal_args(force_regenerate=True))
        assert cfg.training.force_retrain is True

    def test_result_dir_default(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.result_dir == "./experiment_results"

    def test_result_dir_override(self):
        cfg = GenerationConfig(_minimal_args(result_dir="/tmp/results"))
        assert cfg.result_dir == "/tmp/results"


# ---------------------------------------------------------------------------
# get_shot_configs()
# ---------------------------------------------------------------------------

class TestGetShotConfigs:

    def test_default_shot_configs(self):
        cfg = GenerationConfig(_minimal_args())
        assert cfg.get_shot_configs() == [1, 3, 5]

    def test_custom_shot_configs(self):
        cfg = GenerationConfig(_minimal_args(shot_configs=[1, 5, 10, 20]))
        assert cfg.get_shot_configs() == [1, 5, 10, 20]

    def test_single_shot_config(self):
        cfg = GenerationConfig(_minimal_args(shot_configs=[1]))
        assert cfg.get_shot_configs() == [1]

    def test_returns_list(self):
        cfg = GenerationConfig(_minimal_args())
        result = cfg.get_shot_configs()
        assert isinstance(result, list)

    def test_shot_configs_are_ints(self):
        cfg = GenerationConfig(_minimal_args(shot_configs=[1, 3, 5]))
        for shot in cfg.get_shot_configs():
            assert isinstance(shot, int)
