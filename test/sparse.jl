using LinearAlgebra
using SparseArrays
using Random

@testset "Sparse" begin
    # Set up a big diagonal system
    n = 100
    rng = MersenneTwister(123)
    A = randn(rng, n)
    function linear_vel_big!(dx, x, t)
        dx .= 0.0
        for i = 1:n
            dx[i] = -A[i] * x[i]
        end
    end

    function linear_jac_big!(J, x, t)
        J .= 0.0
        for i = 1:n
            J[i, i] = -A[i]
        end
    end

    t0 = 0.0
    t1 = 1.0
    dt = 0.01

    x = ones(n)
    Q = spzeros(n, n)

    exp_x = exp.(-A * (t1 - t0)) .* x
    exp_Q = spdiagm(exp.(-A * (t1 - t0)))

    # This also checks that sharing memory across individual calls of the solvers works
    _state_alloc1 = zeros(n)
    _state_alloc2 = zeros(n)
    _state_alloc3 = zeros(n)
    _state_alloc4 = zeros(n)
    _jac_alloc = spzeros(n, n)
    _mat_alloc1 = spzeros(n, n)
    _mat_alloc2 = spzeros(n, n)
    _mat_alloc3 = spzeros(n, n)
    function reset_state!()
        x .= 1.0
        Q .= 0.0
        for i = 1:n
            Q[i, i] = 1.0
        end
    end

    function test_output()
        # Ensure that Q is still sparse
        @test isa(Q, SparseMatrixCSC)

        # Check the solution - should be diagonal
        # No additional zeros should be introduced
        @test nnz(Q) == count(!iszero, Q) == n

        @test isapprox(x, exp_x, atol = 1e-4)
        for i = 1:n
            @test Q[i, i] ≈ exp_Q[i, i] atol = 1e-6
        end
    end

    @testset "straight rk4" begin
        reset_state!()

        state_eov_rk4!(
            x,
            Q,
            linear_vel_big!,
            linear_jac_big!,
            t0,
            t1,
            dt;
            _state_alloc1 = _state_alloc1,
            _state_alloc2 = _state_alloc2,
            _state_alloc3 = _state_alloc3,
            _mat_alloc1 = _mat_alloc1,
            _mat_alloc2 = _mat_alloc2,
            _mat_alloc3 = _mat_alloc3,
            _jac_alloc = _jac_alloc,
        )

        test_output()

    end

    @testset "rescaling" begin
        reset_state!()

        state_eov_rk4_rescaling!(
            x,
            Q,
            linear_vel_big!,
            linear_jac_big!,
            t0,
            t1,
            dt,
            10.0;
            _state_alloc1 = _state_alloc1,
            _state_alloc2 = _state_alloc2,
            _state_alloc3 = _state_alloc3,
            _mat_alloc1 = _mat_alloc1,
            _mat_alloc2 = _mat_alloc2,
            _mat_alloc3 = _mat_alloc3,
            _jac_alloc = _jac_alloc,
        )

        test_output()
    end

    # Star method is not set up to handle sparseness yet
    # TODO: fix this, somehow. Probably requires the state also being a sparse vector which is difficult to handle.
    # @testset "star" begin
    #     reset_state!()

    #     state_eov_rk4_star!(
    #         x,
    #         Q,
    #         linear_vel_big!,
    #         t0,
    #         t1,
    #         dt,
    #         1e-6;
    #         _state_alloc1 = _state_alloc1,
    #         _state_alloc2 = _state_alloc2,
    #         _state_alloc3 = _state_alloc3,
    #         _state_alloc4 = _state_alloc4,
    #     )

    #     test_output()
    # end

end


@testset "Sparse different" begin
    # setup where the state is dense but the Jacobian is sparse, which is more common in practice
    n = 100
    rng = MersenneTwister(123)
    A = randn(rng, n)
    function linear_vel_big!(dx, x, t)
        dx .= 0.0
        for i = 1:n
            dx[i] = -A[i] * x[i]
        end
    end

    function linear_jac_big!(J, x, t)
        J .= 0.0
        for i = 1:n
            J[i, i] = -A[i]
        end
    end

    t0 = 0.0
    dt = 0.01

    x = ones(n)
    Q = Matrix{Float64}(I, n, n)

    exp_x = exp.(-A * (dt - t0)) .* x
    exp_Q = Diagonal(exp.(-A * (dt - t0)))

    _state_alloc1 = zeros(n)
    _state_alloc2 = zeros(n)
    _state_alloc3 = zeros(n)
    _state_alloc4 = zeros(n)
    _jac_alloc = spzeros(n, n)
    _mat_alloc1 = zeros(n, n)
    _mat_alloc2 = zeros(n, n)
    _mat_alloc3 = zeros(n, n)

    state_eov_rk4_step!(
        x,
        Q,
        linear_vel_big!,
        linear_jac_big!,
        t0,
        dt;
        _state_alloc1 = _state_alloc1,
        _state_alloc2 = _state_alloc2,
        _state_alloc3 = _state_alloc3,
        _mat_alloc1 = _mat_alloc1,
        _mat_alloc2 = _mat_alloc2,
        _mat_alloc3 = _mat_alloc3,
        _jac_alloc = _jac_alloc,
    )

    @test isapprox(x, exp_x, atol = 1e-4)
    for i = 1:n
        @test Q[i, i] ≈ exp_Q[i, i] atol = 1e-6
    end
end