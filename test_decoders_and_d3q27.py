import numpy as np
from quantum_discrete_gaussian import QuantumDiscreteGaussian


def test_get_d3q27_velocity_ordering():
    qdg = QuantumDiscreteGaussian()
    vX, vY, vZ = qdg.get_d3q27_velocity_ordering()
    assert len(vX) == 27 and len(vY) == 27 and len(vZ) == 27

    # Check rest velocity at index 0
    assert vX[0] == 0 and vY[0] == 0 and vZ[0] == 0

    # Check that values are in {-1,0,1}
    assert set(np.unique(vX)).issubset({-1, 0, 1})
    assert set(np.unique(vY)).issubset({-1, 0, 1})
    assert set(np.unique(vZ)).issubset({-1, 0, 1})


def test_compute_3d_probability_distribution_lbm_order():
    qdg = QuantumDiscreteGaussian()
    mu_x, mu_y, mu_z = 0.0, 0.0, 0.0
    sigma_sq = 0.2
    probs_27 = qdg.compute_3d_probability_distribution_lbm_order(mu_x, mu_y, mu_z, sigma_sq)
    assert probs_27.shape == (27,)
    # When mu=0 all directions with same |c| should have same probability per symmetry
    # Sum should be 1
    assert abs(np.sum(probs_27) - 1.0) < 1e-12


def test_convert_quantum_samples_to_lbm_order_roundtrip():
    qdg = QuantumDiscreteGaussian()
    # Simulate deterministic velocity counts: all shots at (0,0,0)
    velocity_counts = {(0, 0, 0): 1000}
    probs = qdg.convert_quantum_samples_to_lbm_order(velocity_counts)
    # All mass should be at index 0
    assert probs[0] == 1.0 and np.sum(probs) == 1.0


if __name__ == '__main__':
    test_get_d3q27_velocity_ordering()
    print('get_d3q27_velocity_ordering: OK')
    test_compute_3d_probability_distribution_lbm_order()
    print('compute_3d_probability_distribution_lbm_order: OK')
    test_convert_quantum_samples_to_lbm_order_roundtrip()
    print('convert_quantum_samples_to_lbm_order_roundtrip: OK')
