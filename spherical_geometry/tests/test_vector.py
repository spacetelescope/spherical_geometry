import pytest

import numpy as np

from spherical_geometry.vector import normalize_vector

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_normalize_vector():
    rng = np.random.default_rng(0)  # Ensure reproducibility for any random operations
    nvec = 17  # >= 6
    # Test normalization of a single vector of shape (3,)
    single = rng.random(3)
    single_normalized = normalize_vector(single)
    assert np.allclose(np.linalg.norm(single_normalized), 1.0)

    # Test normalization of a random vector of shape (N, 3)
    random_2d = rng.random((nvec, 3))
    random_normalized = normalize_vector(random_2d)
    assert np.allclose(np.linalg.norm(random_normalized, axis=1), 1.0)

    # Test normalization of a random vector of shape (N, 3) with a nan or zero
    # vector in the middle of the array. assert vectors that follow are
    # normalized correctly.
    random_2d = rng.random((nvec, 3))
    # null vector:
    random_2d[nvec // 2 - 1] = 0.0
    # nan vector:
    #random_2d[nvec // 2 + 1, 0] = np.nan
    random_2d[2] = np.nan
    mask = np.ones(nvec, dtype=bool)
    mask[nvec // 2 - 1] = False
    #mask[nvec // 2 + 1] = False
    mask[2] = False
    random_normalized = normalize_vector(random_2d)

    assert np.allclose(np.linalg.norm(random_normalized[mask], axis=1), 1.0, rtol=0, atol=1.0e-14)
    # Ensure that the null and nan vectors remain unchanged
    assert np.all(np.all(~np.isfinite(random_normalized[~mask]), axis=1))
