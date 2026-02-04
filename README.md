# EoVSolvers.jl

[![Build Status](https://github.com/liamblake/EoVSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/liamblake/EoVSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)

Given a general nonlinear ODE

$$\frac{\mathrm{d}x}{\mathrm{d}t} = f(x, t),$$

with solution operator $F$ such that $x(t) = F(x_0, t)$, the corresponding (matrix) equation of variations (EoV) is

$$\frac{\mathrm{d}\Psi(x, t)}{\mathrm{d}t} = \nabla f(F(x, t), t) \Psi(x,t),$$

with initial condition $\Psi(x, 0) = I$. This package implements several numerical methods for jointly solving the ODE and its EoV. The methods are all based on the classical 4th-order Runge-Kutta method, and are designed to be memory efficient across many runs (e.g. propagating samples through a system) by reusing several pre-allocated arrays across many steps.


*Github Copilot was used to write the tests in this package and assisted with method documentation.*

## Available methods
- `state_rk4!` - the 4th-order Runge-Kutta method for the ODE only, updating the initial state vector in place.
- `state_eov_rk4!` - the 4th-order Runge-Kutta method for the ODE and its EoV, updating both the initial state vector and the initial matrix condition for the EoV in place.
- `state_eov_rk4_rescaling!` - the 4th-order Runge-Kutta method for the ODE and its EoV, with periodic rescaling of the EoV solution to prevent numerical overflow.
- `state_eov_rk4_star!` - the 4th-order Runge-Kutta method for the ODE and an approximation of the EoV solution with a star finite-difference method. This is typically more stable than solving the EoV directly but is more expensive. This method does not require the Jacobian of the system.


