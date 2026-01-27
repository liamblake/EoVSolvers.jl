# EoVSolvers.jl

[![Build Status](https://github.com/liamblake/EoVSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/liamblake/EoVSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)

Given a general nonlinear ODE
$$
\frac{\mathrm{d}x}{\mathrm{d}t} = f(x, t),
$$
with solution operator \(F\) such that \(x(t) = F(x_0, t)\), the corresponding (matrix) equation of variations (EoV) is
$$
\frac{\mathrm{d}\Psi(x, t)}{\mathrm{d}t} = \nabla f(F(x, t), t) \Psi(x,t),
$$
with initial condition \(\Psi(x, 0) = I\). This package implements several numerical methods for jointly solving the ODE and its EoV.


*Github Copilot was used to write the tests in this package and assisted with method documentation.*