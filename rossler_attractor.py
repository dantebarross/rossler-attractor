import numpy as np

def rossler_ode(state, a, b, c):
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def rk4_step(ode_func, state, dt, *args):
    k1 = ode_func(state, *args) * dt
    k2 = ode_func(state + 0.5 * k1, *args) * dt
    k3 = ode_func(state + 0.5 * k2, *args) * dt
    k4 = ode_func(state + k3, *args) * dt
    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
    return new_state

def simulate_rossler(a=0.2, b=0.2, c=5.7, dt=0.01, num_steps=100000, initial_state=None, method='rk4'):
    if dt <= 0:
        raise ValueError("dt must be positive")
    if max(abs(a), abs(b), abs(c)) > 100:
        dt = min(dt, 1e-4)
    if initial_state is None:
        initial_state = np.array([0.0, 0.0, 0.0])
    if np.all(initial_state == 0.0):
        initial_state = initial_state + 1e-6
    trajectory = np.zeros((num_steps + 1, 3))
    trajectory[0] = initial_state
    current_state = initial_state.copy()
    for i in range(num_steps):
        if method == 'rk4':
            new_state = rk4_step(rossler_ode, current_state, dt, a, b, c)
        elif method == 'euler':
            deriv = rossler_ode(current_state, a, b, c)
            new_state = current_state + deriv * dt
        else:
            raise ValueError(f"Unsupported method: {method}")
        if not np.all(np.isfinite(new_state)):
            new_state = current_state
        current_state = new_state
        trajectory[i+1] = current_state
    return trajectory

def calculate_lyapunov_exponent(traj1, traj2, dt, epsilon=1e-5):
    distances = np.linalg.norm(traj1 - traj2, axis=1)
    # Use the second half of the trajectory to avoid transient effects
    N = len(distances)
    distances = distances[N//2:]
    times = np.arange(len(distances)) * dt
    valid = distances > 0
    if not np.any(valid):
        return 0.0
    log_ratios = np.log(distances[valid] / epsilon)
    slope, _ = np.polyfit(times[valid], log_ratios, 1)
    if slope < 0.05:
        return 0.0
    return slope
