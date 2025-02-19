import pytest
import numpy as np
from rossler_attractor import rossler_ode, rk4_step, simulate_rossler, calculate_lyapunov_exponent

def test_rk4_solver():
    def constant_derivative(state, *args):
        return np.array([1.0])
    state = np.array([0.0])
    dt = 0.1
    new_state = rk4_step(constant_derivative, state, dt)
    assert np.isclose(new_state[0], 0.1)

    def exp_derivative(state, *args):
        return state
    state = np.array([1.0])
    new_state = rk4_step(exp_derivative, state, 0.1)
    expected = 1.0 * (1 + 0.1 + 0.1**2/2 + 0.1**3/6 + 0.1**4/24)
    assert np.isclose(new_state[0], expected, rtol=1e-6)

def test_chaotic_parameters_positive_lyapunov():
    initial_state = np.array([0.1, 0.1, 0.1])
    epsilon = 1e-5
    perturbed_state = initial_state + epsilon * np.ones(3)
    traj1 = simulate_rossler(num_steps=10000, initial_state=initial_state)
    traj2 = simulate_rossler(num_steps=10000, initial_state=perturbed_state)
    lyapunov = calculate_lyapunov_exponent(traj1, traj2, dt=0.01)
    assert lyapunov > 0.0

def test_non_chaotic_c_parameter():
    initial_state = np.array([0.1, 0.1, 0.1])
    epsilon = 1e-5
    perturbed_state = initial_state + epsilon * np.ones(3)
    traj1 = simulate_rossler(c=2.0, num_steps=10000, initial_state=initial_state)
    traj2 = simulate_rossler(c=2.0, num_steps=10000, initial_state=perturbed_state)
    lyapunov = calculate_lyapunov_exponent(traj1, traj2, dt=0.01)
    assert lyapunov <= 0.0

def test_divergence_of_trajectories():
    initial_state = np.array([0.0, 0.0, 0.0])
    epsilon = 1e-5
    traj1 = simulate_rossler(num_steps=1000, initial_state=initial_state)
    traj2 = simulate_rossler(num_steps=1000, initial_state=initial_state + epsilon * np.ones(3))
    final_distance = np.linalg.norm(traj1[-1] - traj2[-1])
    assert final_distance > 1e-5

def test_edge_case_zero_dt():
    with pytest.raises(ValueError):
        simulate_rossler(dt=0.0)

def test_edge_case_large_parameters():
    traj = simulate_rossler(a=1e3, b=1e3, c=1e3, num_steps=100)
    assert np.all(np.isfinite(traj))

def test_negative_parameters():
    traj = simulate_rossler(a=-0.2, b=-0.2, c=-5.7, num_steps=100)
    assert np.all(np.isfinite(traj))

def test_solver_accuracy():
    y0 = np.array([1.0])
    dt = 0.1
    steps = 10
    exact = np.exp(-dt * steps)

    current_rk4 = y0.copy()
    for _ in range(steps):
        current_rk4 = rk4_step(lambda y, *_: -y, current_rk4, dt)
    rk4_error = abs(current_rk4[0] - exact)

    current_euler = y0.copy()
    for _ in range(steps):
        deriv = -current_euler
        current_euler += deriv * dt
    euler_error = abs(current_euler[0] - exact)

    assert rk4_error < euler_error

if __name__ == '__main__':
    traj = simulate_rossler(num_steps=1000)
    print("Final state of simulation:", traj[-1])
    initial_state = np.array([0.1, 0.1, 0.1])
    epsilon = 1e-5
    traj1 = simulate_rossler(num_steps=10000, initial_state=initial_state)
    traj2 = simulate_rossler(num_steps=10000, initial_state=initial_state + epsilon * np.ones(3))
    lyapunov = calculate_lyapunov_exponent(traj1, traj2, dt=0.01)
    print("Calculated Lyapunov exponent:", lyapunov)
