"""
Preprocessing utilities for machine learning (NumPy-only).

This module implements common scaling/normalization routines and dataset
splitting helpers using NumPy only, with robust error handling and
NumPy-style docstrings designed to be testable via doctest.

Functions
---------
standardize
    Z-score standardization (feature-wise).
minmax_scale
    Feature-wise min-max scaling to a given range.
maxabs_scale
    Feature-wise scaling by max absolute value.
l2_normalize_rows
    Row-wise L2 normalization (typical for vector inputs).
train_test_split
    Split arrays or matrices into train and test subsets.
train_val_test_split
    Split arrays or matrices into train, validation, and test subsets.
"""
# TODO Turn this script into a module

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

__all__ = [
    'ArrayLike',
    'standardize',
    'minmax_scale',
    'maxabs_scale',
    'l2_normalize_rows',
    'train_test_split',
    'train_val_test_split',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ---------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------
def _ensure_2d_numeric(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a 2D numeric NumPy array."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if not np.issubdtype(arr.dtype, np.number):
        # Try converting to float explicitly to catch strings/objects.
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _ensure_1d_vector(y: Optional[ArrayLike], name: str = "y") -> Optional[np.ndarray]:
    """Ensure y is a 1D array (numeric), or None."""
    if y is None:
        return None
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")
    # Allow non-numeric labels for stratify (e.g., strings), but for math ops we cast
    return arr


def _check_Xy_shapes(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    if y is not None and len(y) != X.shape[0]:
        raise ValueError(
            f"X and y must have compatible first dimension; got len(y)={len(y)} "
            f"and X.shape[0]={X.shape[0]}."
        )


def _rng_from_seed(random_state: Optional[int]) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if not (isinstance(random_state, (int, np.integer))):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(random_state))


# ---------------------------------------------------------------------
# Scaling / Normalization
# ---------------------------------------------------------------------
def standardize(
    X: ArrayLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Z-score standardization (feature-wise).

    Each feature column is transformed to `(X - mean) / std` when enabled.
    Columns with zero variance are left at 0 when `with_std=True`.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    with_mean : bool, default=True
        Center features by subtracting the column mean.
    with_std : bool, default=True
        Scale features by dividing by the column standard deviation.
    ddof : int, default=0
        Delta degrees of freedom for variance/std calculation.
    return_params : bool, default=False
        If True, also return a dict with keys ``mean`` and ``scale``.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Standardized array.
    params : dict, optional
        Only if `return_params=True`. Contains:
        - 'mean': ndarray of shape (n_features,)
        - 'scale': ndarray of shape (n_features,)

    Raises
    ------
    ValueError
        If X is not 2D or is empty.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1., 2.], [3., 2.], [5., 2.]])
    >>> Z, params = standardize(X, return_params=True)
    >>> Z.mean(axis=0).round(7).tolist()  # centered (second col zero-variance)
    [0.0, 0.0]
    >>> params["scale"].tolist()  # std of second column is 0 -> scale 1 used to avoid divide-by-zero
    [1.632993161855452, 1.0]
    """
    X = _ensure_2d_numeric(X, "X")
    mean = X.mean(axis=0) if with_mean else np.zeros(X.shape[1], dtype=float)
    Xc = X - mean if with_mean else X.copy()

    if with_std:
        # ddof-safe std; for zero std columns, use scale=1 to avoid division by zero
        std = Xc.std(axis=0, ddof=ddof)
        scale = std.copy()
        scale[scale == 0.0] = 1.0
        X_out = Xc / scale
    else:
        scale = np.ones(X.shape[1], dtype=float)
        X_out = Xc

    if return_params:
        return X_out, {"mean": mean, "scale": scale}
    return X_out


def minmax_scale(
    X: ArrayLike,
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Scale features to a specified range (feature-wise).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    feature_range : tuple(float, float), default=(0.0, 1.0)
        Desired value range for each feature.
    return_params : bool, default=False
        If True, also return a dict with keys ``min`` and ``scale``.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Scaled array.
    params : dict, optional
        Only if `return_params=True`. Contains:
        - 'min': ndarray of shape (n_features,)
        - 'scale': ndarray of shape (n_features,) (difference max-min; zero replaced by 1)

    Raises
    ------
    ValueError
        If X is not 2D or empty, or if feature_range is invalid.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0., 10.], [5., 10.], [10., 10.]])
    >>> X2, params = minmax_scale(X, feature_range=(0, 1), return_params=True)
    >>> X2[:, 0].tolist()
    [0.0, 0.5, 1.0]
    >>> X2[:, 1].tolist()  # zero range column -> all mapped to lower bound
    [0.0, 0.0, 0.0]
    >>> params["scale"].tolist()
    [10.0, 1.0]
    """
    X = _ensure_2d_numeric(X, "X")
    if not (
        isinstance(feature_range, tuple)
        and len(feature_range) == 2
        and all(isinstance(v, (int, float)) for v in feature_range)
    ):
        raise ValueError("feature_range must be a tuple of two numeric values (min, max).")
    fr_min, fr_max = float(feature_range[0]), float(feature_range[1])
    if fr_min >= fr_max:
        raise ValueError("feature_range must have min < max.")

    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    range_ = Xmax - Xmin
    scale = range_.copy()
    scale[scale == 0.0] = 1.0  # avoid divide-by-zero

    X01 = (X - Xmin) / scale
    X_out = X01 * (fr_max - fr_min) + fr_min

    if return_params:
        return X_out, {"min": Xmin, "scale": scale, "feature_range": (fr_min, fr_max)}
    return X_out


def maxabs_scale(
    X: ArrayLike,
    *,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Scale features by their maximum absolute value (feature-wise).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    return_params : bool, default=False
        If True, also return a dict with key ``scale`` (= max abs per feature,
        zeros replaced by 1).

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Scaled array.
    params : dict, optional
        Only if `return_params=True`. Contains:
        - 'scale': ndarray of shape (n_features,)

    Raises
    ------
    ValueError
        If X is not 2D or empty.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-2., 0.], [1., 0.], [2., 0.]])
    >>> X2, params = maxabs_scale(X, return_params=True)
    >>> X2[:, 0].tolist()
    [-1.0, 0.5, 1.0]
    >>> X2[:, 1].tolist()  # zero column -> unchanged (div by 1)
    [0.0, 0.0, 0.0]
    >>> params["scale"].tolist()
    [2.0, 1.0]
    """
    X = _ensure_2d_numeric(X, "X")
    maxabs = np.max(np.abs(X), axis=0)
    scale = maxabs.copy()
    scale[scale == 0.0] = 1.0
    X_out = X / scale
    if return_params:
        return X_out, {"scale": scale}
    return X_out


def l2_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization.

    Each row x is replaced by x / max(||x||_2, eps).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    eps : float, default=1e-12
        Floor value to avoid division by zero.

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)
        Row-wise L2-normalized array.

    Raises
    ------
    ValueError
        If X is not 2D or empty, or eps <= 0.
    TypeError
        If X contains non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[3., 4.], [0., 0.]])
    >>> Xn = l2_normalize_rows(X)
    >>> np.allclose(np.linalg.norm(Xn[0]), 1.0)
    True
    >>> Xn[1].tolist()  # zero row stays zero
    [0.0, 0.0]
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    X = _ensure_2d_numeric(X, "X")
    norms = np.linalg.norm(X, axis=1)
    denom = np.maximum(norms, eps)[:, None]
    return X / denom


# ---------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------
def _stratified_indices(y: np.ndarray, test_size: float, rng: np.random.Generator):
    """Return train/test indices with class-wise proportional sampling."""
    # y may be strings/objects; we group by unique values
    classes, y_indices = np.unique(y, return_inverse=True)
    train_idx = []
    test_idx = []
    for cls in range(len(classes)):
        cls_indices = np.flatnonzero(y_indices == cls)
        rng.shuffle(cls_indices)
        n_test = max(1, int(round(test_size * len(cls_indices)))) if len(classes) > 1 else int(
            round(test_size * len(cls_indices))
        )
        n_test = min(n_test, len(cls_indices) - 1) if len(cls_indices) > 1 else n_test
        test_idx.append(cls_indices[:n_test])
        train_idx.append(cls_indices[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.
    y : array_like, shape (n_samples,), optional
        Target vector. If provided, is split in the same way as X.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0 < test_size < 1).
    shuffle : bool, default=True
        Whether to shuffle the data before splitting (ignored when stratify is provided).
    stratify : array_like, optional
        If provided, data is split in a stratified fashion using these labels.
        Must be 1D and have length n_samples.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test : ndarray
        Split feature matrices.
    y_train, y_test : ndarray, optional
        Returned only if `y` is provided.

    Raises
    ------
    ValueError
        If input shapes are invalid, or test_size is not in (0, 1), or stratify has wrong shape.
    TypeError
        If random_state is not an int or None.
    """
    X = _ensure_2d_numeric(X, "X")
    y_arr = _ensure_1d_vector(y, "y")
    _check_Xy_shapes(X, y_arr)
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float in (0, 1).")

    n = X.shape[0]
    rng = _rng_from_seed(random_state)

    if stratify is not None:
        strat = _ensure_1d_vector(stratify, "stratify")
        if len(strat) != n:
            raise ValueError("stratify must have the same length as X.")
        # stratified split
        train_idx, test_idx = _stratified_indices(strat, test_size, rng)
    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        n_test = int(round(test_size * n))
        n_test = min(max(n_test, 1), n - 1) if n > 1 else n_test
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    if y_arr is None:
        return X_train, X_test
    y_train, y_test = y_arr[train_idx], y_arr[test_idx]
    return X_train, X_test, y_train, y_test

def train_val_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Split arrays into train, validation, and test subsets.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.
    y : array_like, shape (n_samples,), optional
        Target vector. If provided, is split in the same way as X.
    val_size : float, default=0.1
        Proportion of the dataset for the validation split (0 < val_size < 1).
    test_size : float, default=0.2
        Proportion of the dataset for the test split (0 < test_size < 1).
    shuffle : bool, default=True
        Whether to shuffle before splitting (ignored when stratify is provided).
    stratify : array_like, optional
        If provided, data is split in a stratified fashion using these labels.
        Must be 1D and have length n_samples.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test : ndarray
        Split feature matrices.
    y_train, y_val, y_test : ndarray, optional
        Returned only if `y` is provided.

    Raises
    ------
    ValueError
        If sizes are invalid or shapes mismatch.
    TypeError
        If random_state is not an int or None.
    """
    if not (0.0 < val_size < 1.0) or not (0.0 < test_size < 1.0):
        raise ValueError("val_size and test_size must be floats in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0.")

    X = _ensure_2d_numeric(X, "X")
    y_arr = _ensure_1d_vector(y, "y")
    _check_Xy_shapes(X, y_arr)

    n = X.shape[0]
    rng = _rng_from_seed(random_state)

    # Helper to bound counts while keeping sets non-empty when possible.
    def _bounded_count(k: int, prop: float) -> int:
        c = int(round(prop * k))
        if k <= 1:
            return 0 if c < k else 1  # degenerate small groups
        return min(max(c, 1), k - 1)

    val_prop_remaining = val_size / (1.0 - test_size)

    if stratify is not None:
        strat = _ensure_1d_vector(stratify, "stratify")
        if len(strat) != n:
            raise ValueError("stratify must have the same length as X.")

        classes, y_idx = np.unique(strat, return_inverse=True)
        train_idx_parts, val_idx_parts, test_idx_parts = [], [], []

        for cls in range(len(classes)):
            cls_indices = np.flatnonzero(y_idx == cls)
            rng.shuffle(cls_indices)

            # test split for this class
            n_test_c = _bounded_count(len(cls_indices), test_size)
            test_c = cls_indices[:n_test_c]
            remaining_c = cls_indices[n_test_c:]

            # val split (relative to remaining)
            n_val_c = _bounded_count(len(remaining_c), val_prop_remaining)
            val_c = remaining_c[:n_val_c]
            train_c = remaining_c[n_val_c:]

            # collect
            test_idx_parts.append(test_c)
            val_idx_parts.append(val_c)
            train_idx_parts.append(train_c)

        train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)
        val_idx   = np.concatenate(val_idx_parts)   if val_idx_parts   else np.array([], dtype=int)
        test_idx  = np.concatenate(test_idx_parts)  if test_idx_parts  else np.array([], dtype=int)

        # keep overall order randomized but reproducible
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        n_test = _bounded_count(n, test_size)
        test_idx = indices[:n_test]
        remaining = indices[n_test:]

        n_val = _bounded_count(len(remaining), val_prop_remaining)
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    if y_arr is None:
        return X_train, X_val, X_test

    y_train, y_val, y_test = y_arr[train_idx], y_arr[val_idx], y_arr[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test
