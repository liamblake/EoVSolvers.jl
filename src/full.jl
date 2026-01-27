
"""
$(TYPEDSIGNATURES)

Iterate the Runge-Kutta 4th order method for a given ODE model from time `t0` to `t1`.

# Arguments
- `x::AbstractVector{X}`: Current state vector. This is used as the initial condition and modified in-place to contain the updated state after the integration.
- `f!::V`: Function that computes the derivative of the state in-place. It should have the signature `f!(dx, x, t)`, where `dx` is the output vector, `x` is the current state, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `t0::T`: Initial time.
- `t1::T`: Final time.
- `dt::T`: Time step size.
- `_alloc1::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_alloc2::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_alloc3::AbstractVector{X}`: Preallocated vector for intermediate computations.

# Returns
Nothing, as the argument `x` is modified in-place to contain the updated state after the integration.

"""
function state_rk4!(
    x::AbstractVector{X},
    f!::V,
    t0::T,
    t1::T,
    dt::T;
    _alloc1::AbstractVector{X} = similar(x),
    _alloc2::AbstractVector{X} = similar(x),
    _alloc3::AbstractVector{X} = similar(x),
) where {X<:Real,V<:Function,T<:Real}
    t = t0
    while t + dt < t1
        state_rk4_step!(x, f!, t, dt; _alloc1 = _alloc1, _alloc2 = _alloc2, _alloc3 = _alloc3)
        t += dt
    end
    if t < t1
        state_rk4_step!(x, f!, t, t1 - t; _alloc1 = _alloc1, _alloc2 = _alloc2, _alloc3 = _alloc3)
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

Iterate the Runge-Kutta 4th order method for the state and state transition matrix of an ODE model.

# Arguments
- `x::AbstractVector{X}`: Current state vector. This is used as the initial condition and modified in-place to contain the updated state after the step.
- `Q::AbstractMatrix{X}`: Current state transition matrix. This is used as the initial condition and modified in-place to contain the updated state transition matrix after the step.
- `f!::V`: Function that computes the derivative of the state and state transition matrix in-place. It should have the signature `f!(dx, dQ, x, Q, t)`, where `dx` is the output vector for the state derivative, `dQ` is the output matrix for the state transition matrix derivative, `x` is the current state, `Q` is the current state transition matrix, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `t0::T`: Current time.
- `dt::T`: Time step size.
- `_alloc1::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_alloc2::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_alloc3::AbstractVector{X}`: Preallocated vector for intermediate computations.
- `_allocQ1::AbstractMatrix{X}`: Preallocated matrix for intermediate computations.
- `_allocQ2::AbstractMatrix{X}`: Preallocated matrix for intermediate computations.
- `_allocQ3::AbstractMatrix{X}`: Preallocated matrix for intermediate computations.

# Returns
Nothing, as the arguments `x` and `Q` are modified in-place to contain the updated state and state transition matrix after the step.

"""
function state_eov_rk4!(
    x::AbstractVector{X},
    Q::AbstractMatrix{X},
    f!::V,
    jac!::J,
    t0::T,
    t1::T,
    dt::T;
    _state_alloc1::AbstractVector{X} = similar(x),
    _state_alloc2::AbstractVector{X} = similar(x),
    _state_alloc3::AbstractVector{X} = similar(x),
    _mat_alloc1::AbstractMatrix{X} = similar(Q),
    _mat_alloc2::AbstractMatrix{X} = similar(Q),
    _mat_alloc3::AbstractMatrix{X} = similar(Q),
) where {X<:Real,V<:Function,J<:Function,T<:Real}
    t = t0
    while t + dt < t1
        state_eov_rk4_step!(
            x,
            Q,
            f!,
            jac!,
            t,
            dt;
            _state_alloc1 = _state_alloc1,
            _state_alloc2 = _state_alloc2,
            _state_alloc3 = _state_alloc3,
            _mat_alloc1 = _mat_alloc1,
            _mat_alloc2 = _mat_alloc2,
            _mat_alloc3 = _mat_alloc3,
        )
        t += dt
    end
    if t < t1
        state_eov_rk4_step!(
            x,
            Q,
            f!,
            jac!,
            t,
            t1 - t;
            _state_alloc1 = _state_alloc1,
            _state_alloc2 = _state_alloc2,
            _state_alloc3 = _state_alloc3,
            _mat_alloc1 = _mat_alloc1,
            _mat_alloc2 = _mat_alloc2,
            _mat_alloc3 = _mat_alloc3,
        )
    end

    return nothing
end