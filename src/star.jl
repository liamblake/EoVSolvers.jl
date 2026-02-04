"""
$(TYPEDSIGNATURES)

Jointly solve an ODE and estimate the spatial gradient using the star grid method. Rather than solving the EoV directly, this method estimates the spatial gradient with a finite difference approximation on a star grid. This is typically more numerically stable in 3-dimensions and higher but is more computationally demaining (requiring 2n + 1 evaluations of the ODE). This is Jacobian-free.

IMPORTANT: this method assumes that the initial condition for the state transition matrix is the identity matrix. Anything in Q when the function is called will be overwritten without being read.
"""
function state_eov_rk4_star!(
    x::AbstractVector{X},
    Q::AbstractMatrix{X},
    f!::V,
    t0::T,
    t1::T,
    dt::T,
    delta::Vector{X};
    _state_alloc1::AbstractVector{X} = similar(x),
    _state_alloc2::AbstractVector{X} = similar(x),
    _state_alloc3::AbstractVector{X} = similar(x),
    _state_alloc4::AbstractVector{X} = similar(x),
) where {X<:Real,V<:Function,T<:Real}

    N = length(x)
    if length(delta) != N
        throw(ArgumentError("Delta vector must have the same length as the state vector x."))
    end

    for n = 1:N
        # Positive perturbation
        copyto!(_state_alloc4, x)
        _state_alloc4[n] += delta[n]
        state_rk4!(
            _state_alloc4,
            f!,
            t0,
            t1,
            dt;
            _alloc1 = _state_alloc1,
            _alloc2 = _state_alloc2,
            _alloc3 = _state_alloc3,
        )
        Q[:, n] = _state_alloc4

        # Negative perturbation
        copyto!(_state_alloc4, x)
        _state_alloc4[n] -= delta[n]
        state_rk4!(
            _state_alloc4,
            f!,
            t0,
            t1,
            dt;
            _alloc1 = _state_alloc1,
            _alloc2 = _state_alloc2,
            _alloc3 = _state_alloc3,
        )
        Q[:, n] .-= _state_alloc4
        rmul!(@view(Q[:, n]), 1 / (2 * abs(delta[n])))
    end

    # Propagate original point as normal
    state_rk4!(
        x,
        f!,
        t0,
        t1,
        dt;
        _alloc1 = _state_alloc1,
        _alloc2 = _state_alloc2,
        _alloc3 = _state_alloc3,
    )

    return nothing
end


function state_eov_rk4_star!(
    x::AbstractVector{X},
    Q::AbstractMatrix{X},
    f!::V,
    t0::T,
    t1::T,
    dt::T,
    delta::X;
    _state_alloc1::AbstractVector{X} = similar(x),
    _state_alloc2::AbstractVector{X} = similar(x),
    _state_alloc3::AbstractVector{X} = similar(x),
    _state_alloc4::AbstractVector{X} = similar(x),
) where {X<:Real,V<:Function,T<:Real}
    delta_vec = fill(delta, length(x))
    return state_eov_rk4_star!(
        x,
        Q,
        f!,
        t0,
        t1,
        dt,
        delta_vec;
        _state_alloc1 = _state_alloc1,
        _state_alloc2 = _state_alloc2,
        _state_alloc3 = _state_alloc3,
        _state_alloc4 = _state_alloc4,
    )

end