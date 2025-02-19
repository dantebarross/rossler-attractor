# Rössler Attractor Simulation

This repository contains a Python implementation of the **Rössler attractor**, a chaotic dynamical system defined by a set of nonlinear differential equations. The simulation uses the Runge-Kutta 4th order (RK4) method to solve the equations and calculates the Lyapunov exponent to verify chaotic behavior.

## Features
- **Simulation:** Implements the Rössler attractor using RK4 and Euler methods.
- **Chaos Analysis:** Calculates the Lyapunov exponent to determine chaotic properties.
- **Unit Testing:** Contains comprehensive tests using `pytest` to validate solver accuracy, chaotic properties, and edge cases.
- **Command-Line Output:** Running the tests directly prints the final simulation state and the calculated Lyapunov exponent.

## Installation
Ensure you have Python 3.12 or later installed. Install required packages using:
```bash
pip install numpy pytest
```

## Usage
- **Run Simulation and See Output:**
  ```bash
  python test_rossler_attractor.py
  ```
  This prints the final state of the simulation and the calculated Lyapunov exponent.

- **Run Unit Tests:**
  ```bash
  pytest test_rossler_attractor.py -v
  ```

## Repository Structure
- `rossler_attractor.py`: Contains the implementation of the Rössler attractor, the RK4 solver, and functions for chaos analysis.
- `test_rossler_attractor.py`: Contains unit tests for validating the implementation.
- `README.md`: This documentation file.

## Git Commands for Deployment
The repository is configured to be pushed to GitHub. To set it up, run:
```bash
git remote add origin https://github.com/dantebarross/rossler-attractor.git
git branch -M main
git add .
git commit -m "Organize repo and add README"
git push -u origin main
```

## License
This project is licensed under the MIT License.
