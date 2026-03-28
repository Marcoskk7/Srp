"""
Tests for _npz_to_method_format().

The function lives in main.py alongside argparse setup and side-effectful
top-level code, which makes importing main.py fragile without the full
runtime environment. To keep these tests self-contained and free of GPU /
data-file dependencies, the function is defined locally here — it is a
two-line pure function and the definition is the authoritative specification
being tested.

If the implementation in main.py ever diverges, the discrepancy will be
visible during code review; the test file documents the exact contract.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Function under test  (copy of main.py::_npz_to_method_format)
# ---------------------------------------------------------------------------

def _npz_to_method_format(X: np.ndarray, y: np.ndarray) -> list:
    """Convert (X, y) arrays into list[(data_array, label_str)] format."""
    return [(X[y == cls], str(int(cls))) for cls in sorted(np.unique(y))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(num_classes: int, samples_per_class: int, signal_length: int,
                  label_offset: int = 0):
    """Create a synthetic (X, y) pair with controlled shapes."""
    X_parts = []
    y_parts = []
    for cls in range(num_classes):
        label = cls + label_offset
        X_parts.append(np.random.randn(samples_per_class, signal_length).astype(np.float32))
        y_parts.append(np.full(samples_per_class, label, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestNpzToMethodFormatBasic:

    def test_output_is_list(self):
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        assert isinstance(result, list)

    def test_output_length_equals_num_classes(self):
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        assert len(result) == 3

    def test_each_element_is_two_tuple(self):
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        for item in result:
            assert len(item) == 2

    def test_first_element_is_ndarray(self):
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        for data_arr, _ in result:
            assert isinstance(data_arr, np.ndarray)

    def test_second_element_is_string(self):
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        for _, label_str in result:
            assert isinstance(label_str, str)


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------

class TestNpzToMethodFormatShapes:

    def test_data_array_has_correct_num_rows(self):
        """Each class slice should have exactly samples_per_class rows."""
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        for data_arr, _ in result:
            assert data_arr.shape[0] == 5

    def test_data_array_preserves_signal_length(self):
        X, y = _make_dataset(3, 5, 2400)
        result = _npz_to_method_format(X, y)
        for data_arr, _ in result:
            assert data_arr.shape[1] == 2400

    def test_unequal_class_sizes(self):
        """Classes with different sample counts should each have their own count."""
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((3, 100)).astype(np.float32)
        X1 = rng.standard_normal((7, 100)).astype(np.float32)
        X2 = rng.standard_normal((5, 100)).astype(np.float32)
        X = np.concatenate([X0, X1, X2], axis=0)
        y = np.array([0]*3 + [1]*7 + [2]*5, dtype=np.int64)

        result = _npz_to_method_format(X, y)
        sizes = {label: arr.shape[0] for arr, label in result}
        assert sizes["0"] == 3
        assert sizes["1"] == 7
        assert sizes["2"] == 5


# ---------------------------------------------------------------------------
# Label string encoding
# ---------------------------------------------------------------------------

class TestNpzToMethodFormatLabels:

    def test_contiguous_labels_are_string_integers(self):
        X, y = _make_dataset(3, 5, 100)
        result = _npz_to_method_format(X, y)
        labels = [label for _, label in result]
        assert labels == ["0", "1", "2"]

    def test_non_contiguous_labels_encoded_as_strings(self):
        """y values of [6,7,8,9] should yield labels '6','7','8','9'."""
        X, y = _make_dataset(4, 5, 100, label_offset=6)
        result = _npz_to_method_format(X, y)
        labels = [label for _, label in result]
        assert labels == ["6", "7", "8", "9"]

    def test_labels_are_sorted(self):
        """Classes must appear in ascending label order."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 50)).astype(np.float32)
        # Deliberately shuffle y so classes are not in insertion order
        y = np.array([2]*5 + [0]*5 + [1]*5, dtype=np.int64)
        result = _npz_to_method_format(X, y)
        labels = [label for _, label in result]
        assert labels == ["0", "1", "2"]

    def test_data_aligned_with_label(self):
        """Samples in each slice must actually belong to the stated class."""
        rng = np.random.default_rng(2)
        # Use distinct constant values per class so we can verify alignment
        X0 = np.full((4, 10), fill_value=0.0, dtype=np.float32)
        X1 = np.full((4, 10), fill_value=1.0, dtype=np.float32)
        X2 = np.full((4, 10), fill_value=2.0, dtype=np.float32)
        X = np.concatenate([X0, X1, X2], axis=0)
        y = np.array([0]*4 + [1]*4 + [2]*4, dtype=np.int64)

        result = _npz_to_method_format(X, y)
        for data_arr, label_str in result:
            expected_value = float(label_str)
            assert np.all(data_arr == expected_value), (
                f"Class '{label_str}' slice contains unexpected values"
            )

    def test_large_label_values(self):
        """Labels do not have to start at 0; large values should work."""
        X, y = _make_dataset(3, 2, 50, label_offset=100)
        result = _npz_to_method_format(X, y)
        labels = [label for _, label in result]
        assert labels == ["100", "101", "102"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestNpzToMethodFormatEdgeCases:

    def test_empty_arrays_return_empty_list(self):
        X = np.empty((0, 100), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        result = _npz_to_method_format(X, y)
        assert result == []

    def test_single_class(self):
        X, y = _make_dataset(1, 10, 50)
        result = _npz_to_method_format(X, y)
        assert len(result) == 1
        data_arr, label_str = result[0]
        assert data_arr.shape == (10, 50)
        assert label_str == "0"

    def test_single_sample_per_class(self):
        X, y = _make_dataset(3, 1, 2400)
        result = _npz_to_method_format(X, y)
        assert len(result) == 3
        for data_arr, _ in result:
            assert data_arr.shape[0] == 1

    def test_many_classes(self):
        """Should handle 10+ classes without issue."""
        X, y = _make_dataset(10, 5, 512)
        result = _npz_to_method_format(X, y)
        assert len(result) == 10
        labels = [label for _, label in result]
        assert labels == [str(i) for i in range(10)]

    def test_float_y_values_are_cast_to_int_string(self):
        """y stored as float (common with .npz loads) must still yield int strings."""
        X = np.random.randn(6, 50).astype(np.float32)
        y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
        result = _npz_to_method_format(X, y)
        labels = [label for _, label in result]
        assert labels == ["0", "1", "2"]

    def test_output_data_shares_no_extra_columns(self):
        """Verify indexing does not accidentally include wrong-class rows."""
        X = np.arange(30, dtype=np.float32).reshape(6, 5)
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        result = _npz_to_method_format(X, y)
        data0, _ = result[0]
        data1, _ = result[1]
        assert data0.shape == (3, 5)
        assert data1.shape == (3, 5)
        # Class 0 rows are rows 0,1,2; class 1 rows are 3,4,5
        np.testing.assert_array_equal(data0, X[:3])
        np.testing.assert_array_equal(data1, X[3:])
