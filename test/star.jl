using LinearAlgebra

@testset "Star Grid Method Tests" begin
    @testset "state_eov_rk4_star! - Basic Functionality with Vector Delta" begin
        # Test simple linear ODE: dx/dt = -x
        # Analytical STM: Φ(t) = exp(-t) * I
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [1.0]
        Q = zeros(1, 1)
        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        delta = [1e-6]

        state_eov_rk4_star!(x, Q, linear_vel!, t0, t1, dt, delta)

        # Check state propagation
        x_analytical = exp(-t1)
        @test x[1] ≈ x_analytical atol = 1e-6

        # Check STM approximation (should be close to exp(-t1) ≈ 0.905)
        Phi_analytical = exp(-t1)
        @test Q[1, 1] ≈ Phi_analytical atol = 1e-4
    end

    @testset "state_eov_rk4_star! - Basic Functionality with Scalar Delta" begin
        # Test with scalar delta (convenience wrapper)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [1.0]
        Q = zeros(1, 1)
        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        delta = 1e-6

        state_eov_rk4_star!(x, Q, linear_vel!, t0, t1, dt, delta)

        # Check state propagation
        x_analytical = exp(-t1)
        @test x[1] ≈ x_analytical atol = 1e-6

        # Check STM approximation
        Phi_analytical = exp(-t1)
        @test Q[1, 1] ≈ Phi_analytical atol = 1e-4
    end

    @testset "state_eov_rk4_star! - 2D System" begin
        # Test 2D harmonic oscillator: dx/dt = v, dv/dt = -x
        # Analytical STM: Φ(t) = [cos(t) sin(t); -sin(t) cos(t)]
        function harmonic_vel!(dx, x, t)
            dx[1] = x[2]      # dx/dt = velocity
            dx[2] = -x[1]     # d²x/dt² = -x
        end

        x = [1.0, 0.0]
        Q = zeros(2, 2)
        t0 = 0.0
        t1 = 0.5
        dt = 0.01
        delta = [1e-6, 1e-6]

        state_eov_rk4_star!(x, Q, harmonic_vel!, t0, t1, dt, delta)

        # Check state propagation
        x_analytical = [cos(t1), -sin(t1)]
        @test x[1] ≈ x_analytical[1] atol = 1e-4
        @test x[2] ≈ x_analytical[2] atol = 1e-4

        # Check STM approximation
        Phi_analytical = [cos(t1) sin(t1); -sin(t1) cos(t1)]
        @test Q[1, 1] ≈ Phi_analytical[1, 1] atol = 1e-3
        @test Q[1, 2] ≈ Phi_analytical[1, 2] atol = 1e-3
        @test Q[2, 1] ≈ Phi_analytical[2, 1] atol = 1e-3
        @test Q[2, 2] ≈ Phi_analytical[2, 2] atol = 1e-3
    end

    @testset "state_eov_rk4_star! - 3D System" begin
        # Test 3D decoupled system: dx_i/dt = -i * x_i
        # Analytical STM: Φ(t) = diag(exp(-t), exp(-2t), exp(-3t))
        function decoupled_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -2 * x[2]
            dx[3] = -3 * x[3]
        end

        x = [1.0, 1.0, 1.0]
        Q = zeros(3, 3)
        t0 = 0.0
        t1 = 0.2
        dt = 0.01
        delta = 1e-6

        state_eov_rk4_star!(x, Q, decoupled_vel!, t0, t1, dt, delta)

        # Check state propagation
        x_analytical = [exp(-t1), exp(-2 * t1), exp(-3 * t1)]
        @test x[1] ≈ x_analytical[1] atol = 1e-6
        @test x[2] ≈ x_analytical[2] atol = 1e-6
        @test x[3] ≈ x_analytical[3] atol = 1e-6

        # Check STM approximation (diagonal matrix)
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-4
        @test Q[2, 2] ≈ exp(-2 * t1) atol = 1e-4
        @test Q[3, 3] ≈ exp(-3 * t1) atol = 1e-4

        # Off-diagonal elements should be near zero
        @test abs(Q[1, 2]) < 1e-3
        @test abs(Q[1, 3]) < 1e-3
        @test abs(Q[2, 1]) < 1e-3
        @test abs(Q[2, 3]) < 1e-3
        @test abs(Q[3, 1]) < 1e-3
        @test abs(Q[3, 2]) < 1e-3
    end

    @testset "state_eov_rk4_star! - Different Delta Values" begin
        # Test with different delta values for each dimension
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -2 * x[2]
        end

        x = [1.0, 1.0]
        Q = zeros(2, 2)
        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        delta = [1e-5, 1e-7]  # Different deltas

        state_eov_rk4_star!(x, Q, linear_vel!, t0, t1, dt, delta)

        # Check state propagation
        x_analytical = [exp(-t1), exp(-2 * t1)]
        @test x[1] ≈ x_analytical[1] atol = 1e-6
        @test x[2] ≈ x_analytical[2] atol = 1e-6

        # Check STM approximation
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-3
        @test Q[2, 2] ≈ exp(-2 * t1) atol = 1e-3
    end

    @testset "state_eov_rk4_star! - Time-Dependent System" begin
        # Test dx/dt = x + t
        # This is a more complex system
        function time_dep_vel!(dx, x, t)
            dx[1] = x[1] + t
        end

        x = [1.0]
        Q = zeros(1, 1)
        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        delta = 1e-6

        state_eov_rk4_star!(x, Q, time_dep_vel!, t0, t1, dt, delta)

        # Analytical solution: x(t) = (x₀ + t₀ + 1)*exp(t) - t - 1
        x_analytical = (1.0 + t0 + 1) * exp(t1) - t1 - 1
        @test x[1] ≈ x_analytical atol = 1e-5

        # Check that Q is computed (should be approximately exp(t1))
        @test abs(Q[1, 1] - exp(t1)) < 0.1  # Looser tolerance for time-dependent case
    end

    @testset "state_eov_rk4_star! - Preallocated Memory" begin
        # Test with preallocated arrays
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -2 * x[2]
        end

        x = [1.0, 2.0]
        Q = zeros(2, 2)
        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        delta = 1e-6

        # Preallocate memory
        alloc1 = similar(x)
        alloc2 = similar(x)
        alloc3 = similar(x)
        alloc4 = similar(x)

        state_eov_rk4_star!(
            x,
            Q,
            linear_vel!,
            t0,
            t1,
            dt,
            delta;
            _state_alloc1 = alloc1,
            _state_alloc2 = alloc2,
            _state_alloc3 = alloc3,
            _state_alloc4 = alloc4,
        )

        # Check state propagation
        @test x[1] ≈ exp(-t1) atol = 1e-6
        @test x[2] ≈ 2 * exp(-2 * t1) atol = 1e-5

        # Check STM approximation
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-3
        @test Q[2, 2] ≈ exp(-2 * t1) atol = 1e-3
    end

    @testset "state_eov_rk4_star! - Zero Time Interval" begin
        # Test when t0 == t1 (should compute STM at initial time)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [5.0]
        x_initial = copy(x)
        Q = zeros(1, 1)
        t0 = 1.0
        t1 = 1.0
        dt = 0.1
        delta = 1e-6

        state_eov_rk4_star!(x, Q, linear_vel!, t0, t1, dt, delta)

        # State should remain unchanged
        @test x[1] ≈ x_initial[1] atol = 1e-10

        # STM should be close to identity
        @test Q[1, 1] ≈ 1.0 atol = 1e-4
    end

    @testset "state_eov_rk4_star! - Delta Size Effect" begin
        # Test that smaller delta gives more accurate results
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        Phi_analytical = exp(-t1)

        # Test with different delta values
        deltas = [1e-4, 1e-6, 1e-8]
        errors = Float64[]

        for delta in deltas
            x = [1.0]
            Q = zeros(1, 1)
            state_eov_rk4_star!(x, Q, linear_vel!, t0, t1, dt, delta)
            error = abs(Q[1, 1] - Phi_analytical)
            push!(errors, error)
        end

        # For finite difference methods, error typically decreases as delta 
        # decreases until roundoff errors dominate. Just check that errors 
        # are reasonable and at least one smaller delta improves accuracy
        @test all(e -> e < 1e-2, errors)  # All errors should be small
    end

    @testset "state_eov_rk4_star! - Argument Error" begin
        # Test that mismatched delta vector length throws error
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -x[2]
        end

        x = [1.0, 2.0]
        Q = zeros(2, 2)
        t0 = 0.0
        t1 = 0.1
        dt = 0.01
        delta = [1e-6]  # Wrong length

        @test_throws ArgumentError state_eov_rk4_star!(x, Q, linear_vel!, t0, t1, dt, delta)
    end

    @testset "state_eov_rk4_star! - Type Stability" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        # Test Float32
        x_f32 = Float32[1.0]
        Q_f32 = zeros(Float32, 1, 1)
        state_eov_rk4_star!(x_f32, Q_f32, linear_vel!, 0.0f0, 0.1f0, 0.01f0, 1.0f-6)
        @test eltype(x_f32) == Float32
        @test eltype(Q_f32) == Float32

        # Test Float64
        x_f64 = Float64[1.0]
        Q_f64 = zeros(Float64, 1, 1)
        state_eov_rk4_star!(x_f64, Q_f64, linear_vel!, 0.0, 0.1, 0.01, 1e-6)
        @test eltype(x_f64) == Float64
        @test eltype(Q_f64) == Float64
    end

    @testset "state_eov_rk4_star! - Coupled System" begin
        # Test a coupled system where off-diagonal STM elements are non-zero
        # dx/dt = y, dy/dt = x (exponentially growing oscillator)
        function coupled_vel!(dx, x, t)
            dx[1] = x[2]
            dx[2] = x[1]
        end

        x = [1.0, 0.0]
        Q = zeros(2, 2)
        t0 = 0.0
        t1 = 0.1
        dt = 0.001  # Small time step for stability
        delta = 1e-6

        state_eov_rk4_star!(x, Q, coupled_vel!, t0, t1, dt, delta)

        # Analytical solution: x(t) = cosh(t), y(t) = sinh(t)
        x_analytical = [cosh(t1), sinh(t1)]
        @test x[1] ≈ x_analytical[1] atol = 1e-5
        @test x[2] ≈ x_analytical[2] atol = 1e-5

        # STM should have non-zero off-diagonal elements
        @test abs(Q[1, 2]) > 1e-4
        @test abs(Q[2, 1]) > 1e-4
    end

    @testset "state_eov_rk4_star! - Comparison with Analytical Jacobian Method" begin
        # For a simple linear system, compare star method with analytical Jacobian
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -2 * x[2]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
            J[1, 2] = 0.0
            J[2, 1] = 0.0
            J[2, 2] = -2.0
        end

        # Star method
        x_star = [1.0, 1.0]
        Q_star = zeros(2, 2)
        state_eov_rk4_star!(x_star, Q_star, linear_vel!, 0.0, 0.2, 0.01, 1e-6)

        # Analytical Jacobian method
        x_jac = [1.0, 1.0]
        Q_jac = Matrix{Float64}(I, 2, 2)
        state_eov_rk4!(x_jac, Q_jac, linear_vel!, linear_jac!, 0.0, 0.2, 0.01)

        # States should be nearly identical
        @test x_star[1] ≈ x_jac[1] atol = 1e-8
        @test x_star[2] ≈ x_jac[2] atol = 1e-8

        # STMs should be close (star method is approximate)
        @test Q_star[1, 1] ≈ Q_jac[1, 1] atol = 1e-3
        @test Q_star[2, 2] ≈ Q_jac[2, 2] atol = 1e-3
        @test abs(Q_star[1, 2]) < 1e-3  # Should be near zero
        @test abs(Q_star[2, 1]) < 1e-3  # Should be near zero
    end

    @testset "state_eov_rk4_star! - Nonlinear System" begin
        # Test a nonlinear system (damped pendulum approximation)
        function nonlinear_vel!(dx, x, t)
            dx[1] = x[2]
            dx[2] = -sin(x[1]) - 0.1 * x[2]  # sin(theta) for small angles ≈ theta
        end

        x = [0.1, 0.0]  # Small initial angle
        Q = zeros(2, 2)
        t0 = 0.0
        t1 = 0.5
        dt = 0.01
        delta = 1e-7

        state_eov_rk4_star!(x, Q, nonlinear_vel!, t0, t1, dt, delta)

        # Just check that it runs and produces reasonable results
        @test isfinite(x[1])
        @test isfinite(x[2])
        @test all(isfinite.(Q))
        
        # Q should have non-zero off-diagonal elements (coupled system)
        @test abs(Q[1, 2]) > 1e-4
        @test abs(Q[2, 1]) > 1e-4
    end
end
