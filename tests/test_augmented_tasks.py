"""
Tests for augmentation support in FinetuneTask, MetaTask, and BaseTrainer.

Covers:
- FinetuneTask: aug_data / augment_num params
- MetaTask: aug_data / augment_num params
- BaseTrainer._get_augment_num()
- _prepare_aug_data() helper (logic extracted for isolated testing)

RED phase: these tests are written BEFORE implementation, so they will fail
initially against the unmodified data_loader.py / base_trainer.py.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader import FinetuneTask, MetaTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_folders(num_classes: int, samples_per_class: int, signal_len: int = 50):
    """Create list[(data_array, label_str)] as expected by task classes."""
    return [
        (np.random.randn(samples_per_class, signal_len).astype(np.float32), str(i))
        for i in range(num_classes)
    ]


def _make_aug_data(num_classes: int, aug_per_class: int, signal_len: int = 50):
    """Create a parallel list of augmented arrays (one ndarray per class)."""
    return [
        np.random.randn(aug_per_class, signal_len).astype(np.float32)
        for _ in range(num_classes)
    ]


# ===========================================================================
# FinetuneTask – backward-compatibility (no augmentation)
# ===========================================================================

class TestFinetuneTaskBackwardCompat:
    """Existing behavior must be preserved when aug_data is not provided."""

    def test_support_size_without_augmentation(self):
        folders = _make_folders(4, 20)
        task = FinetuneTask(folders, support_num=3, seed=0)
        # 4 classes × 3 support = 12
        assert len(task.support_files) == 12

    def test_query_size_without_augmentation(self):
        folders = _make_folders(4, 20)
        task = FinetuneTask(folders, support_num=3, seed=0)
        # 4 classes × (20-3) query = 68
        assert len(task.query_files) == 68

    def test_support_labels_without_augmentation(self):
        folders = _make_folders(3, 10)
        task = FinetuneTask(folders, support_num=2, seed=0)
        assert len(task.support_labels) == 6  # 3×2

    def test_query_labels_without_augmentation(self):
        folders = _make_folders(3, 10)
        task = FinetuneTask(folders, support_num=2, seed=0)
        assert len(task.query_labels) == 24  # 3×8


# ===========================================================================
# FinetuneTask – with augmentation
# ===========================================================================

class TestFinetuneTaskAugmentation:

    def test_support_set_grows_by_augment_num(self):
        """Support set should have (support_num + augment_num) × num_classes items."""
        folders = _make_folders(4, 20)
        aug_data = _make_aug_data(4, 10)
        task = FinetuneTask(folders, support_num=2, seed=0,
                            aug_data=aug_data, augment_num=3)
        # 4 classes × (2 real + 3 aug) = 20
        assert len(task.support_files) == 20

    def test_query_set_unchanged_with_augmentation(self):
        """Query set must contain only real samples; augmented data must NOT appear."""
        folders = _make_folders(4, 20)
        aug_data = _make_aug_data(4, 10)
        task = FinetuneTask(folders, support_num=2, seed=0,
                            aug_data=aug_data, augment_num=5)
        # query = 4 × (20-2) = 72  (unchanged)
        assert len(task.query_files) == 72

    def test_support_labels_include_augmented_entries(self):
        """Support labels list must account for augmented samples."""
        folders = _make_folders(3, 15)
        aug_data = _make_aug_data(3, 10)
        task = FinetuneTask(folders, support_num=2, seed=0,
                            aug_data=aug_data, augment_num=4)
        # 3 × (2 + 4) = 18
        assert len(task.support_labels) == 18

    def test_query_labels_unchanged_with_augmentation(self):
        folders = _make_folders(3, 15)
        aug_data = _make_aug_data(3, 10)
        task = FinetuneTask(folders, support_num=2, seed=0,
                            aug_data=aug_data, augment_num=4)
        # 3 × (15 - 2) = 39
        assert len(task.query_labels) == 39

    def test_augment_num_capped_by_available_aug_samples(self):
        """When augment_num > len(aug_data[cls]), use all available aug samples."""
        folders = _make_folders(2, 20)
        # Only 3 aug samples per class, but augment_num=10
        aug_data = _make_aug_data(2, 3)
        task = FinetuneTask(folders, support_num=1, seed=0,
                            aug_data=aug_data, augment_num=10)
        # 2 × (1 real + 3 aug) = 8
        assert len(task.support_files) == 8

    def test_none_aug_class_skips_augmentation(self):
        """If aug_data[i] is None, that class gets no augmentation."""
        folders = _make_folders(3, 20)
        aug_data = [
            np.random.randn(5, 50).astype(np.float32),
            None,
            np.random.randn(5, 50).astype(np.float32),
        ]
        task = FinetuneTask(folders, support_num=2, seed=0,
                            aug_data=aug_data, augment_num=3)
        # class 0: 2+3=5, class 1: 2+0=2, class 2: 2+3=5  → total 12
        assert len(task.support_files) == 12

    def test_augment_num_zero_no_change(self):
        """augment_num=0 must produce identical result to no augmentation."""
        folders = _make_folders(3, 10)
        aug_data = _make_aug_data(3, 5)
        task_aug = FinetuneTask(folders, support_num=2, seed=0,
                                aug_data=aug_data, augment_num=0)
        task_plain = FinetuneTask(folders, support_num=2, seed=0)
        assert len(task_aug.support_files) == len(task_plain.support_files)
        assert len(task_aug.query_files) == len(task_plain.query_files)

    def test_aug_data_none_no_change(self):
        """aug_data=None must produce identical result to no augmentation."""
        folders = _make_folders(3, 10)
        task_aug = FinetuneTask(folders, support_num=2, seed=0,
                                aug_data=None, augment_num=3)
        task_plain = FinetuneTask(folders, support_num=2, seed=0)
        assert len(task_aug.support_files) == len(task_plain.support_files)

    def test_support_and_query_labels_are_consistent(self):
        """Every label value must appear in both support and query."""
        folders = _make_folders(4, 15)
        aug_data = _make_aug_data(4, 5)
        task = FinetuneTask(folders, support_num=2, seed=42,
                            aug_data=aug_data, augment_num=2)
        support_label_set = set(task.support_labels)
        query_label_set = set(task.query_labels)
        assert support_label_set == query_label_set == {0, 1, 2, 3}

    def test_augmented_samples_come_from_aug_data(self):
        """Augmented support samples should contain rows from aug_data, not real data."""
        # Use distinct fill values so we can identify the origin
        folders = [
            (np.full((10, 5), fill_value=float(i), dtype=np.float32), str(i))
            for i in range(2)
        ]
        # Augmented samples use fill_value 99 and 88
        aug_data = [
            np.full((5, 5), fill_value=99.0, dtype=np.float32),
            np.full((5, 5), fill_value=88.0, dtype=np.float32),
        ]
        task = FinetuneTask(folders, support_num=1, seed=0,
                            aug_data=aug_data, augment_num=2)
        # For class 0: 1 real sample (value=0) + 2 aug samples (value=99)
        # For class 1: 1 real sample (value=1) + 2 aug samples (value=88)
        support_arr = np.array(task.support_files)  # (6, 5)
        assert len(support_arr) == 6
        # Check that aug samples are present
        assert np.any(support_arr == 99.0)
        assert np.any(support_arr == 88.0)
        # Query must not contain aug samples
        query_arr = np.array(task.query_files)
        assert not np.any(query_arr == 99.0)
        assert not np.any(query_arr == 88.0)


# ===========================================================================
# MetaTask – backward-compatibility (no augmentation)
# ===========================================================================

class TestMetaTaskBackwardCompat:
    """Existing MetaTask behavior must be preserved when no augmentation is given."""

    def test_support_size(self):
        folders = _make_folders(6, 30)
        task = MetaTask(folders, num_classes=3, support_num=2,
                        query_num=5, seed=0)
        assert len(task.support_files) == 6   # 3 × 2

    def test_query_size(self):
        folders = _make_folders(6, 30)
        task = MetaTask(folders, num_classes=3, support_num=2,
                        query_num=5, seed=0)
        assert len(task.query_files) == 15   # 3 × 5


# ===========================================================================
# MetaTask – with augmentation
# ===========================================================================

class TestMetaTaskAugmentation:

    def test_support_set_grows_by_augment_num(self):
        folders = _make_folders(6, 30)
        aug_data = _make_aug_data(6, 10)
        task = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                        seed=0, aug_data=aug_data, augment_num=3)
        # 3 sampled classes × (2 real + 3 aug) = 15
        assert len(task.support_files) == 15

    def test_query_unchanged_with_augmentation(self):
        folders = _make_folders(6, 30)
        aug_data = _make_aug_data(6, 10)
        task = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                        seed=0, aug_data=aug_data, augment_num=3)
        # query = 3 × 5 = 15
        assert len(task.query_files) == 15

    def test_augment_num_zero_no_change(self):
        folders = _make_folders(6, 30)
        aug_data = _make_aug_data(6, 10)
        task_aug = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                            seed=7, aug_data=aug_data, augment_num=0)
        task_plain = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                              seed=7)
        assert len(task_aug.support_files) == len(task_plain.support_files)
        assert len(task_aug.query_files) == len(task_plain.query_files)

    def test_aug_data_none_no_change(self):
        folders = _make_folders(6, 30)
        task_aug = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                            seed=7, aug_data=None, augment_num=3)
        task_plain = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                              seed=7)
        assert len(task_aug.support_files) == len(task_plain.support_files)

    def test_augment_num_capped_by_available(self):
        """When augment_num > available aug samples, use all available."""
        folders = _make_folders(6, 30)
        aug_data = _make_aug_data(6, 2)   # only 2 aug samples per class
        task = MetaTask(folders, num_classes=3, support_num=2, query_num=5,
                        seed=0, aug_data=aug_data, augment_num=10)
        # 3 × (2 + 2) = 12
        assert len(task.support_files) == 12


# ===========================================================================
# BaseTrainer._get_augment_num()
# ===========================================================================

class TestGetAugmentNum:
    """Tests for the BaseTrainer._get_augment_num helper."""

    @pytest.fixture
    def trainer_cls(self):
        """Import BaseTrainer — will fail until base_trainer.py is updated."""
        from methods.base_trainer import BaseTrainer

        class _MinimalConfig:
            class training:
                device = "cpu"
                random_seed = 0
            class data:
                dataset = "CWRU"
                task_type = "condition"
                cwru_data_type = "time"
                cwru_fault_source_codes = ["N", "F"]
                cwru_fault_target_codes = ["G"]
            class model:
                feature_dim = 64
                cwru_adaptive_pool_size = 64
                adaptive_pool_size = 64
            def get_shot_configs(self):
                return [1, 3, 5]

            @property
            def effective_signal_length(self):
                return 2400

        # Concrete stub that satisfies abstractmethods
        class _StubTrainer(BaseTrainer):
            def __init__(self, config):
                super().__init__("stub", config)
            def train(self, metatrain_data):
                return None, 0.0
            def test(self, model, metatest_data):
                return {}

        cfg = _MinimalConfig()
        cfg.data.effective_signal_length = 2400
        return _StubTrainer(cfg)

    def test_augment_shot_zero_returns_zero(self, trainer_cls):
        trainer = trainer_cls
        trainer.augment_shot = 0
        assert trainer._get_augment_num(1) == 0
        assert trainer._get_augment_num(5) == 0

    def test_augment_shot_greater_than_shot_returns_difference(self, trainer_cls):
        trainer = trainer_cls
        trainer.augment_shot = 5
        assert trainer._get_augment_num(1) == 4
        assert trainer._get_augment_num(3) == 2

    def test_augment_shot_equal_to_shot_returns_zero(self, trainer_cls):
        trainer = trainer_cls
        trainer.augment_shot = 5
        assert trainer._get_augment_num(5) == 0

    def test_augment_shot_less_than_shot_returns_zero(self, trainer_cls):
        trainer = trainer_cls
        trainer.augment_shot = 3
        assert trainer._get_augment_num(5) == 0


# ===========================================================================
# _prepare_aug_data() logic (standalone, not testing main.py imports)
# ===========================================================================

def _prepare_aug_data_logic(augment_type, metatest_data, Xg, yg,
                             y_target_offset, augment_shot, noise_level):
    """
    Pure function that mirrors the logic in main._prepare_aug_data().
    Defined here locally to keep tests import-free of main.py.
    """
    if augment_type == 'none':
        return []

    aug_data = []
    for local_idx, (real_data, _) in enumerate(metatest_data):
        if augment_type == 'gan':
            if Xg is None or yg is None:
                aug_data.append(None)
                continue
            global_cls = local_idx + y_target_offset
            mask = yg == global_cls
            aug_data.append(Xg[mask] if mask.any() else None)
        elif augment_type == 'noise':
            # generate a noise pool from the real data
            n_pool = augment_shot * 20
            std = np.std(real_data) * noise_level
            noise = np.random.normal(0, std, (n_pool,) + real_data.shape[1:]).astype(np.float32)
            base = real_data[np.random.choice(len(real_data), n_pool, replace=True)]
            aug_data.append(base + noise)
        else:
            aug_data.append(None)
    return aug_data


class TestPrepareAugData:

    def _make_metatest(self, num_classes=4, n_real=20, sig_len=50):
        return [
            (np.random.randn(n_real, sig_len).astype(np.float32), str(i))
            for i in range(num_classes)
        ]

    def test_none_returns_empty_list(self):
        meta = self._make_metatest()
        result = _prepare_aug_data_logic('none', meta, None, None, 0, 5, 0.05)
        assert result == []

    def test_gan_returns_list_of_correct_length(self):
        meta = self._make_metatest(4)
        # GAN samples for global labels 6-9
        Xg = np.random.randn(40, 50).astype(np.float32)
        yg = np.repeat(np.arange(6, 10), 10)
        result = _prepare_aug_data_logic('gan', meta, Xg, yg,
                                         y_target_offset=6, augment_shot=5,
                                         noise_level=0.05)
        assert len(result) == 4

    def test_gan_each_class_has_correct_samples(self):
        meta = self._make_metatest(4)
        Xg = np.random.randn(40, 50).astype(np.float32)
        yg = np.repeat(np.arange(6, 10), 10)  # 10 samples per class
        result = _prepare_aug_data_logic('gan', meta, Xg, yg,
                                         y_target_offset=6, augment_shot=5,
                                         noise_level=0.05)
        for arr in result:
            assert arr is not None
            assert arr.shape[0] == 10

    def test_gan_none_xg_yields_none_entries(self):
        meta = self._make_metatest(3)
        result = _prepare_aug_data_logic('gan', meta, None, None,
                                         y_target_offset=0, augment_shot=5,
                                         noise_level=0.05)
        assert all(v is None for v in result)

    def test_noise_returns_list_of_correct_length(self):
        meta = self._make_metatest(4)
        result = _prepare_aug_data_logic('noise', meta, None, None,
                                          y_target_offset=0, augment_shot=5,
                                          noise_level=0.05)
        assert len(result) == 4

    def test_noise_pool_size_equals_augment_shot_times_20(self):
        meta = self._make_metatest(3)
        result = _prepare_aug_data_logic('noise', meta, None, None,
                                          y_target_offset=0, augment_shot=5,
                                          noise_level=0.05)
        for arr in result:
            assert arr is not None
            assert arr.shape[0] == 100  # 5 * 20

    def test_noise_samples_have_correct_signal_length(self):
        meta = self._make_metatest(2, sig_len=2400)
        result = _prepare_aug_data_logic('noise', meta, None, None,
                                          y_target_offset=0, augment_shot=3,
                                          noise_level=0.05)
        for arr in result:
            assert arr.shape[1] == 2400


# ===========================================================================
# BalancedSampler / get_meta_loader — augmented support indexing
# ===========================================================================

class TestBalancedSamplerWithAugmentation:
    """BalancedSampler must produce correct indices when augmented support
    set has more samples per class than the original shot count."""

    def test_sampler_indices_cover_all_augmented_samples(self):
        """With shot=1 + augment_num=4, sampler indices for each class
        should span the correct range within a 20-element (4cls × 5/cls) array."""
        from data_loader import BalancedSampler
        num_classes = 4
        shot = 1
        augment_num = 4
        total_per_class = shot + augment_num  # 5

        sampler = BalancedSampler(
            num_per_class=total_per_class,
            num_classes=num_classes,
            num_instances=total_per_class,  # this is what the fix should pass
            shuffle=False,
        )
        indices = list(sampler)
        # Should have 4 × 5 = 20 unique indices in [0, 19]
        assert len(indices) == 20
        assert set(indices) == set(range(20))

    def test_sampler_with_wrong_num_instances_produces_overlapping_indices(self):
        """Demonstrates the bug: if num_instances=shot (not shot+augment_num),
        indices from different classes overlap, scrambling class assignments."""
        from data_loader import BalancedSampler
        num_classes = 4
        shot = 1
        augment_num = 4
        total_per_class = shot + augment_num

        # BUG: num_instances=shot instead of total_per_class
        sampler = BalancedSampler(
            num_per_class=total_per_class,
            num_classes=num_classes,
            num_instances=shot,  # wrong!
            shuffle=False,
        )
        indices = list(sampler)
        # With num_instances=1, class j produces [0+j*1..4+j*1]
        # Class 0: [0,1,2,3,4], Class 1: [1,2,3,4,5] — overlap!
        assert len(indices) != len(set(indices)), \
            "Bug demonstration: overlapping indices should exist"


class TestGetMetaLoaderAugmentedSupport:
    """get_meta_loader must handle augmented support sets correctly."""

    def test_support_loader_returns_correct_batch_size_with_augmentation(self):
        """When MetaTask support has augmented samples, the loader batch
        should contain all (shot + augment_num) × num_classes samples."""
        from data_loader import MetaTask, get_meta_loader
        num_classes = 4
        shot = 1
        augment_num = 4
        folders = _make_folders(num_classes, 30)
        aug_data = _make_aug_data(num_classes, 20)

        task = MetaTask(
            folders, num_classes=num_classes,
            support_num=shot, query_num=10, seed=42,
            aug_data=aug_data, augment_num=augment_num,
        )
        support_loader = get_meta_loader(
            task, num_per_class=shot + augment_num,
            split="support", shuffle=False,
            data_type="time", signal_length=50,
        )
        batch_x, batch_y = next(iter(support_loader))
        # Should have 4 × 5 = 20 samples
        assert batch_x.shape[0] == num_classes * (shot + augment_num)

    def test_support_loader_labels_are_class_correct(self):
        """Each class should get exactly (shot + augment_num) labels in the batch."""
        from data_loader import MetaTask, get_meta_loader
        import torch
        num_classes = 4
        shot = 1
        augment_num = 4
        # Use distinct fill values per class to verify correctness
        folders = [
            (np.full((30, 50), fill_value=float(i), dtype=np.float32), str(i))
            for i in range(num_classes)
        ]
        aug_data = [
            np.full((20, 50), fill_value=float(i) + 0.5, dtype=np.float32)
            for i in range(num_classes)
        ]

        task = MetaTask(
            folders, num_classes=num_classes,
            support_num=shot, query_num=10, seed=42,
            aug_data=aug_data, augment_num=augment_num,
        )
        support_loader = get_meta_loader(
            task, num_per_class=shot + augment_num,
            split="support", shuffle=False,
            data_type="time", signal_length=50,
        )
        batch_x, batch_y = next(iter(support_loader))
        # Each class should appear exactly 5 times
        for cls in range(num_classes):
            assert (batch_y == cls).sum().item() == shot + augment_num, \
                f"Class {cls} should have {shot + augment_num} samples, " \
                f"got {(batch_y == cls).sum().item()}"


# ===========================================================================
# MRN view() dimension test
# ===========================================================================

class TestMRNSupportReshape:
    """The MRN support feature reshape must use shot + augment_num, not just shot."""

    def test_view_with_augmented_support(self):
        """Simulates the MRN support_features.view() call with augmented data.
        With shot=1 + augment_num=4, the view should use 5, not 1."""
        import torch
        num_classes = 4
        shot = 1
        augment_num = 4
        feature_dim = 64
        pool_size = 1  # after adaptive pooling

        total_per_class = shot + augment_num
        total_samples = num_classes * total_per_class
        # Simulate feature encoder output: [batch, feature_dim, pool_size]
        support_features = torch.randn(total_samples, feature_dim, pool_size)

        # This is the CORRECT reshape (should work)
        reshaped = support_features.view(
            num_classes, total_per_class, feature_dim, -1
        )
        assert reshaped.shape == (num_classes, total_per_class, feature_dim, pool_size)

        # Take mean across samples per class → class prototypes
        prototypes = reshaped.mean(dim=1)
        assert prototypes.shape == (num_classes, feature_dim, pool_size)

    def test_view_with_wrong_shot_produces_wrong_shape(self):
        """Demonstrates the bug: view(num_classes, shot, ..., -1) silently
        produces a wrong shape instead of raising (the -1 absorbs extra elements)."""
        import torch
        num_classes = 4
        shot = 1
        augment_num = 4
        feature_dim = 64
        pool_size = 1

        total_per_class = shot + augment_num
        total_samples = num_classes * total_per_class
        support_features = torch.randn(total_samples, feature_dim, pool_size)

        # BUG: view(4, 1, 64, -1) → shape (4, 1, 64, 5) instead of (4, 5, 64, 1)
        wrong = support_features.view(num_classes, shot, feature_dim, -1)
        correct = support_features.view(num_classes, total_per_class, feature_dim, -1)
        assert wrong.shape != correct.shape, \
            "Wrong view should have different shape than correct view"
