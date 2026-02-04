"""
$(TYPEDSIGNATURES)

Jointly solve an ODE and the corresponding EoV using a Runge-Kutta 4th order method and adaptive rescaling of the EoV solution to maintain numerical stability.

# Arguments
- `x::AbstractVector{X}`: Current state vector. This is used as the initial condition and modified in-place to contain the updated state after the step.
- `Q::AbstractMatrix{X}`: Current state transition matrix. This is used as the initial condition and modified in-place to contain the updated state transition matrix after the step.
- `f!::V`: Function that computes the derivative of the state in-place. It should have the signature `f!(dx, x, t)`, where `dx` is the output vector, `x` is the current state, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `jac!::J`: Function that computes the Jacobian matrix of the system in-place. It should have the signature `jac!(J, x, t)`, where `J` is the output matrix, `x` is the current state, and `t` is the current time. If the function signature is incorrect, the resulting error will not be wrapped.
- `t0::T`: Initial time.
- `t1::T`: Final time.
- `dt::T`: Time step size.
- `L::Real`: Rescaling threshold. When the norm of the state transition matrix exceeds this value, it will be rescaled to maintain numerical stability. The larger this number, the more efficient the algorithm but the less stable. Setting L = 1 means that the solution is rescaled at every dt step.
- `_state_alloc1::AbstractVector{X}`: Preallocated vector for intermediate state computations.
- `_state_alloc2::AbstractVector{X}`: Preallocated vector for intermediate state computations.
- `_state_alloc3::AbstractVector{X}`: Preallocated vector for intermediate state computations.
- `_mat_alloc1::AbstractMatrix{X}`: Preallocated matrix for intermediate matrix computations.
- `_mat_alloc2::AbstractMatrix{X}`: Preallocated matrix for intermediate matrix computations.
- `_mat_alloc3::AbstractMatrix{X}`: Preallocated matrix for intermediate matrix computations

# Returns
Nothing, as the arguments `x` and `Q` are modified in-place to contain the updated state and state transition matrix after the step.

"""
function state_eov_rk4_rescaling!(
    x::AbstractVector{X},
    Q::AbstractMatrix{X},
    f!::V,
    jac!::J,
    t0::T,
    t1::T,
    dt::T,
    L::X;
    _state_alloc1::AbstractVector{X} = similar(x),
    _state_alloc2::AbstractVector{X} = similar(x),
    _state_alloc3::AbstractVector{X} = similar(x),
    _mat_alloc1::AbstractMatrix{X} = similar(Q),
    _mat_alloc2::AbstractMatrix{X} = similar(Q),
    _mat_alloc3::AbstractMatrix{X} = similar(Q),
    _mat_alloc4::AbstractMatrix{X} = similar(Q),
) where {X<:Real,V<:Function,J<:Function,T<:Real}
    if L < 1
        throw(ArgumentError("Rescaling threshold L must be at least 1 for algorithm to iterate."))
    end

    s = X(1) # Final time scaling - accumulates the normalisation factors
    t = t0

    while t < t1
        nv = opnorm(Q, 1)
        s *= nv
        rmul!(Q, nv^(-1))
        nv = 1.0

        while nv <= L && t < t1
            if t + dt > t1
                dt = t1 - t
            end

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
                _mat_alloc4 = _mat_alloc4,
            )
            t += dt
            nv = opnorm(Q, 1)
        end

        # s *= nv
    end
    # Undo normalisation scalings
    rmul!(Q, s)

    return nothing
end