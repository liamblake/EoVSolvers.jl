using LinearAlgebra

"""
$TYPEDSIGNATURES

Iterate a single step of the Runge-Kutta 4th order method for a given ODE model.

# Arguments
- `x::AbstractVector{X}`: Current state vector. This is used as the initial condition and modified in-place to contain the updated state after the step.
- `f!::V`: Function that computes the derivative of the state in-place. It should have the signature `f!(dx, x, t)`, where `dx` is the output vector, `x` is the current state, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `t0::T`: Current time.
- `dt::T`: Time step size.
- `_alloc1::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_alloc2::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_alloc3::AbstractVector{X}`: Preallocated vector for intermediate computations.

# Returns
Nothing, as the argument `x` is modified in-place to contain the updated state after the step.

"""
function rk4_step!(
    x::AbstractVector{X},
    f!::V,
    t0::T,
    dt::T;
    _alloc1::AbstractVector{X} = similar(x),
    _alloc2::AbstractVector{X} = similar(x),
    _alloc3::AbstractVector{X} = similar(x),
) where {X<:Real,V<:Function,T<:Real}
    # Copy state into state_alloc2
    copyto!(_alloc1, x)

    # Stage 1
    f!(_alloc2, _alloc1, t0)
    rmul!(_alloc2, dt)
    axpy!(1.0 / 6.0, _alloc2, x)

    # Stage 2
    axpby!(1.0, _alloc1, 0.5, _alloc2)
    f!(_alloc3, _alloc2, t0 + 0.5 * dt)
    rmul!(_alloc3, dt)
    axpy!(1.0 / 3.0, _alloc3, x)

    # Stage 3
    axpby!(1.0, _alloc1, 0.5, _alloc3)
    f!(_alloc2, _alloc3, t0 + 0.5 * dt)
    rmul!(_alloc2, dt)
    axpy!(1.0 / 3.0, _alloc2, x)

    # Stage 4
    axpby!(1.0, _alloc1, 1.0, _alloc2)
    f!(_alloc3, _alloc2, t0 + dt)
    rmul!(_alloc3, dt)
    axpy!(1.0 / 6.0, _alloc3, x)

    return nothing
end

"""
$TYPEDSIGNATURES

Iterate a single step of the Runge-Kutta 4th order method for the state and equation of variations of a model, simultaneously.

# Arguments
- `x::AbstractVector{X}`: Current state vector. This is used as the initial condition and modified in-place to contain the updated state after the step.
- `Q::AbstractMatrix{X}`: Current state transition matrix. This is used as the initial condition and modified in-place to contain the updated state transition matrix after the step.
- `f!::V`: Function that computes the derivative of the state in-place. It should have the signature `f!(dx, x, t)`, where `dx` is the output vector, `x` is the current state, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `jac!::J`: Function that computes the Jacobian matrix of the system in-place. It should have the signature `jac!(J, x, t)`, where `J` is the output matrix, `x` is the current state, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `t1::T`: Current time.
- `dt::T`: Time step size.
- `_state_alloc1::AbstractVector{X}`: Preallocated vector for intermediate state computations.
- `_state_alloc2::AbstractVector{X}`: Preallocated vector for intermediate state computations.
- `_state_alloc3::AbstractVector{X}`: Preallocated vector for intermediate state computations.
- `_mat_alloc1::AbstractMatrix{X}`: Preallocated matrix for intermediate matrix computations.
- `_mat_alloc2::AbstractMatrix{X}`: Preallocated matrix for intermediate matrix computations.
- `_mat_alloc3::AbstractMatrix{X}`: Preallocated matrix for intermediate matrix computations.

# Returns
Nothing, as the arguments `x` and `Q` are modified in-place to contain the updated state and state transition matrix after the step.

"""
function rk4_step_state_eov!(
    x::AbstractVector{X},
    Q::AbstractMatrix{X},
    f!::V,
    jac!::J,
    t1::T,
    dt::T;
    _state_alloc1::AbstractVector{X} = similar(x),
    _state_alloc2::AbstractVector{X} = similar(x),
    _state_alloc3::AbstractVector{X} = similar(x),
    _mat_alloc1::AbstractMatrix{X} = similar(Q),
    _mat_alloc2::AbstractMatrix{X} = similar(Q),
    _mat_alloc3::AbstractMatrix{X} = similar(Q),
) where {X<:Real,V<:Function,J<:Function,T<:Real}

    # Copy state into state_alloc2
    copyto!(_state_alloc1, x)
    copyto!(_mat_alloc1, Q)

    # Stage 1
    jac!(_mat_alloc3, _state_alloc1, t1)
    f!(_state_alloc2, _state_alloc1, t1)
    rmul!(_state_alloc2, dt)

    mul!(_mat_alloc2, _mat_alloc3, _mat_alloc1)
    rmul!(_mat_alloc2, dt)

    axpy!(1.0 / 6.0, _state_alloc2, x)
    axpy!(1.0 / 6.0, _mat_alloc2, Q)

    # Stage 2
    # Use in-place computation to avoid allocation
    copyto!(_state_alloc3, _state_alloc1)
    axpy!(0.5, _state_alloc2, _state_alloc3)
    jac!(_mat_alloc3, _state_alloc3, t1 + 0.5 * dt)
    f!(_state_alloc2, _state_alloc3, t1 + 0.5 * dt)
    rmul!(_state_alloc2, dt)

    _mat_alloc2 .= _mat_alloc3 * (_mat_alloc1 + 0.5 * _mat_alloc2)
    rmul!(_mat_alloc2, dt)

    axpy!(1.0 / 3.0, _state_alloc2, x)
    axpy!(1.0 / 3.0, _mat_alloc2, Q)

    # Stage 3
    copyto!(_state_alloc3, _state_alloc1)
    axpy!(0.5, _state_alloc2, _state_alloc3)
    jac!(_mat_alloc3, _state_alloc3, t1 + 0.5 * dt)
    f!(_state_alloc2, _state_alloc3, t1 + 0.5 * dt)
    rmul!(_state_alloc2, dt)

    _mat_alloc2 .= _mat_alloc3 * (_mat_alloc1 + 0.5 * _mat_alloc2)
    rmul!(_mat_alloc2, dt)

    axpy!(1.0 / 3.0, _state_alloc2, x)
    axpy!(1.0 / 3.0, _mat_alloc2, Q)

    # Stage 4
    copyto!(_state_alloc3, _state_alloc1)
    axpy!(1.0, _state_alloc2, _state_alloc3)
    jac!(_mat_alloc3, _state_alloc3, t1 + dt)
    f!(_state_alloc2, _state_alloc3, t1 + dt)
    rmul!(_state_alloc2, dt)

    _mat_alloc2 .= _mat_alloc3 * (_mat_alloc1 + _mat_alloc2)
    rmul!(_mat_alloc2, dt)

    axpy!(1.0 / 6.0, _state_alloc2, x)
    axpy!(1.0 / 6.0, _mat_alloc2, Q)

    return nothing
end
