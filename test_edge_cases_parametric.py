import numpy as np
from quantum_discrete_gaussian import QuantumDiscreteGaussian


def test_parametric_p_bounds():
    qdg = QuantumDiscreteGaussian()

    # Case p = mu^2 + sigma_sq = 0 -> should raise ValueError in parametric circuit
    try:
        qdg.create_quantum_circuit_parametric(0.0, 0.0)
        raise AssertionError("Expected ValueError for p=0, but none was raised")
    except ValueError as e:
        print('p=0 raised ValueError as expected:', str(e))

    # Case p >= 1 -> should raise ValueError
    try:
        qdg.create_quantum_circuit_parametric(1.0, 0.0)  # mu^2 =1 -> p=1
        raise AssertionError("Expected ValueError for p=1, but none was raised")
    except ValueError as e:
        print('p=1 raised ValueError as expected:', str(e))


def test_clamp_and_small_sigma():
    qdg = QuantumDiscreteGaussian()

    # Very small sigma (near zero) with mu=0.0 - valid and should not raise
    mu = 0.0
    sigma_sq = 1e-12
    qc = qdg.create_quantum_circuit_parametric(mu, sigma_sq)
    assert qc is not None
    print('tiny-sigma parametric circuit (mu=0) created successfully')

    # Check _clamp01 behavior
    assert qdg._clamp01(-0.5) == 0.0
    assert qdg._clamp01(1.5) == 1.0
    assert qdg._clamp01(float('nan')) == 0.0
    assert qdg._clamp01(0.5) == 0.5
    print('clamp behavior OK')


if __name__ == '__main__':
    test_parametric_p_bounds()
    test_clamp_and_small_sigma()
    print('Edge-case tests completed')
