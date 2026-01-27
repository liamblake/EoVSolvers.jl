using LinearAlgebra

@testset "Full Integration Tests" begin
    @testset "state_rk4! - Basic Functionality" begin
        # Test simple linear ODE: dx/dt = -x
        # Analytical solution: x(t) = x₀ * exp(-t)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [1.0]
        t0 = 0.0
        t1 = 1.0
        dt = 0.1

        x_initial = copy(x)
        state_rk4!(x, linear_vel!, t0, t1, dt)

        # Analytical solution at t = 1.0
        x_analytical = x_initial[1] * exp(-t1)

        @test length(x) == 1
        @test x[1] ≈ x_analytical atol = 1e-6
    end

    @testset "state_rk4! - 2D Harmonic Oscillator" begin
        # Test 2D harmonic oscillator: d²x/dt² = -x
        # State: [x, dx/dt], dynamics: [dx/dt, -x]
        function harmonic_vel!(dx, x, t)
            dx[1] = x[2]      # dx/dt = velocity
            dx[2] = -x[1]     # d²x/dt² = -x
        end

        # Initial condition: x=1, v=0
        x = [1.0, 0.0]
        t0 = 0.0
        t1 = 2.0
        dt = 0.01

        state_rk4!(x, harmonic_vel!, t0, t1, dt)

        # Analytical solution: x(t) = cos(t), v(t) = -sin(t)
        x_analytical = [cos(t1), -sin(t1)]

        @test x[1] ≈ x_analytical[1] atol = 1e-4
        @test x[2] ≈ x_analytical[2] atol = 1e-4
    end

    @testset "state_rk4! - Time-Dependent System" begin
        # Test dx/dt = t (solution: x(t) = x₀ + t²/2 - t₀²/2)
        function time_dep_vel!(dx, x, t)
            dx[1] = t
        end

        x = [1.0]
        t0 = 0.5
        t1 = 2.0
        dt = 0.1

        state_rk4!(x, time_dep_vel!, t0, t1, dt)

        # Analytical solution: x(t) = 1 + (t₁²/2 - t₀²/2)
        x_analytical = 1.0 + (t1^2 / 2 - t0^2 / 2)

        @test x[1] ≈ x_analytical atol = 1e-6
    end

    @testset "state_rk4! - Partial Final Step" begin
        # Test that partial final step is handled correctly
        # Integrate from t=0 to t=0.25 with dt=0.1
        # Should do 2 full steps (0.0->0.1->0.2) and one partial (0.2->0.25)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [1.0]
        t0 = 0.0
        t1 = 0.25
        dt = 0.1

        state_rk4!(x, linear_vel!, t0, t1, dt)

        # Analytical solution at t = 0.25
        x_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-6
    end

    @testset "state_rk4! - Exact Step Multiple" begin
        # Test when t1 - t0 is exact multiple of dt
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [1.0]
        t0 = 0.0
        t1 = 1.0
        dt = 0.1  # 10 steps exactly

        state_rk4!(x, linear_vel!, t0, t1, dt)

        x_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-6
    end

    @testset "state_rk4! - Preallocated Memory" begin
        # Test with preallocated arrays
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -2 * x[2]
        end

        x = [1.0, 2.0]
        t0 = 0.0
        t1 = 1.0
        dt = 0.1

        # Preallocate memory
        alloc1 = similar(x)
        alloc2 = similar(x)
        alloc3 = similar(x)

        state_rk4!(x, linear_vel!, t0, t1, dt; _alloc1 = alloc1, _alloc2 = alloc2, _alloc3 = alloc3)

        @test x[1] ≈ exp(-t1) atol = 1e-6
        @test x[2] ≈ 2 * exp(-2 * t1) atol = 1e-5  # Slightly relaxed tolerance for 2D system
    end

    @testset "state_rk4! - Zero Integration Interval" begin
        # Test when t0 == t1 (should return unchanged)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [5.0]
        x_initial = copy(x)
        t0 = 1.0
        t1 = 1.0
        dt = 0.1

        state_rk4!(x, linear_vel!, t0, t1, dt)

        @test x[1] ≈ x_initial[1] atol = 1e-12
    end

    @testset "state_rk4! - Small Integration Interval" begin
        # Test when t1 - t0 < dt (single partial step)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        x = [1.0]
        t0 = 0.0
        t1 = 0.05  # Less than dt
        dt = 0.1

        state_rk4!(x, linear_vel!, t0, t1, dt)

        x_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-8
    end

    @testset "state_rk4! - Type Stability" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        # Test Float32
        x_f32 = Float32[1.0]
        state_rk4!(x_f32, linear_vel!, 0.0f0, 1.0f0, 0.1f0)
        @test eltype(x_f32) == Float32

        # Test Float64
        x_f64 = Float64[1.0]
        state_rk4!(x_f64, linear_vel!, 0.0, 1.0, 0.1)
        @test eltype(x_f64) == Float64
    end

    @testset "state_rk4! - Nonlinear System" begin
        # Test Van der Pol oscillator approximation (simple harmonic)
        function oscillator_vel!(dx, x, t)
            dx[1] = x[2]
            dx[2] = -x[1]
        end

        x = [1.0, 0.0]
        t0 = 0.0
        t1 = 2π  # One full period
        dt = 0.01

        # Energy should be approximately conserved
        initial_energy = 0.5 * (x[1]^2 + x[2]^2)

        state_rk4!(x, oscillator_vel!, t0, t1, dt)

        final_energy = 0.5 * (x[1]^2 + x[2]^2)

        @test abs(final_energy - initial_energy) < 0.01

        # After full period, should return close to initial condition
        @test x[1] ≈ 1.0 atol = 0.01
        @test x[2] ≈ 0.0 atol = 0.01
    end

    @testset "state_rk4! - Consistency Across Step Sizes" begin
        # Finer step size should give more accurate result
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        t0 = 0.0
        t1 = 1.0

        # Coarse step
        x_coarse = [1.0]
        state_rk4!(x_coarse, linear_vel!, t0, t1, 0.5)

        # Fine step
        x_fine = [1.0]
        state_rk4!(x_fine, linear_vel!, t0, t1, 0.01)

        x_analytical = exp(-t1)

        # Fine step should be more accurate
        error_coarse = abs(x_coarse[1] - x_analytical)
        error_fine = abs(x_fine[1] - x_analytical)

        @test error_fine < error_coarse
    end

    @testset "state_eov_rk4! - Basic Functionality" begin
        # Test simple linear ODE: dx/dt = -x with state transition matrix
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        x = [1.0]
        Q = Matrix{Float64}(I, 1, 1)
        t0 = 0.0
        t1 = 1.0
        dt = 0.1

        x_initial = copy(x)
        Q_initial = copy(Q)

        state_eov_rk4!(x, Q, linear_vel!, linear_jac!, t0, t1, dt)

        # Analytical solutions
        x_analytical = x_initial[1] * exp(-t1)
        Q_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-6
        @test Q[1, 1] ≈ Q_analytical atol = 1e-6
    end

    @testset "state_eov_rk4! - 2D System" begin
        # Test 2D linear system: dx/dt = Ax where A = [-1 0; 0 -2]
        function linear2d_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = -2 * x[2]
        end

        function linear2d_jac!(J, x, t)
            J[1, 1] = -1.0
            J[1, 2] = 0.0
            J[2, 1] = 0.0
            J[2, 2] = -2.0
        end

        x = [1.0, 2.0]
        Q = Matrix{Float64}(I, 2, 2)
        t0 = 0.0
        t1 = 1.0
        dt = 0.05

        state_eov_rk4!(x, Q, linear2d_vel!, linear2d_jac!, t0, t1, dt)

        # Analytical solutions
        @test x[1] ≈ exp(-t1) atol = 1e-5
        @test x[2] ≈ 2 * exp(-2 * t1) atol = 1e-5

        # Q should be diagonal for this decoupled system
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-5
        @test Q[2, 2] ≈ exp(-2 * t1) atol = 1e-5
        @test abs(Q[1, 2]) < 1e-10
        @test abs(Q[2, 1]) < 1e-10
    end

    @testset "state_eov_rk4! - Partial Final Step" begin
        # Test that partial final step is handled correctly
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        x = [1.0]
        Q = Matrix{Float64}(I, 1, 1)
        t0 = 0.0
        t1 = 0.25  # Not a multiple of dt
        dt = 0.1

        state_eov_rk4!(x, Q, linear_vel!, linear_jac!, t0, t1, dt)

        x_analytical = exp(-t1)
        Q_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-6
        @test Q[1, 1] ≈ Q_analytical atol = 1e-6
    end

    @testset "state_eov_rk4! - Preallocated Memory" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        x = [1.0]
        Q = Matrix{Float64}(I, 1, 1)
        t0 = 0.0
        t1 = 1.0
        dt = 0.1

        # Preallocate memory
        alloc1 = similar(x)
        alloc2 = similar(x)
        alloc3 = similar(x)
        allocQ1 = similar(Q)
        allocQ2 = similar(Q)
        allocQ3 = similar(Q)

        state_eov_rk4!(
            x,
            Q,
            linear_vel!,
            linear_jac!,
            t0,
            t1,
            dt;
            _state_alloc1 = alloc1,
            _state_alloc2 = alloc2,
            _state_alloc3 = alloc3,
            _mat_alloc1 = allocQ1,
            _mat_alloc2 = allocQ2,
            _mat_alloc3 = allocQ3,
        )

        @test x[1] ≈ exp(-t1) atol = 1e-6
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-6
    end

    @testset "state_eov_rk4! - Zero Integration Interval" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        x = [5.0]
        Q = Matrix{Float64}(I, 1, 1)
        x_initial = copy(x)
        Q_initial = copy(Q)
        t0 = 1.0
        t1 = 1.0
        dt = 0.1

        state_eov_rk4!(x, Q, linear_vel!, linear_jac!, t0, t1, dt)

        @test x[1] ≈ x_initial[1] atol = 1e-12
        @test Q[1, 1] ≈ Q_initial[1, 1] atol = 1e-12
    end

    @testset "state_eov_rk4! - Harmonic Oscillator" begin
        # Harmonic oscillator: dx/dt = v, dv/dt = -x
        # Jacobian: J = [0 1; -1 0]
        function harmonic_vel!(dx, x, t)
            dx[1] = x[2]
            dx[2] = -x[1]
        end

        function harmonic_jac!(J, x, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -1.0
            J[2, 2] = 0.0
        end

        x = [1.0, 0.0]
        Q = Matrix{Float64}(I, 2, 2)
        t0 = 0.0
        t1 = π / 2  # Quarter period
        dt = 0.01

        state_eov_rk4!(x, Q, harmonic_vel!, harmonic_jac!, t0, t1, dt)

        # At t = π/2: x(t) = cos(t) = 0, v(t) = -sin(t) = -1
        @test x[1] ≈ 0.0 atol = 1e-3
        @test x[2] ≈ -1.0 atol = 1e-3

        # State transition matrix at t = π/2
        # Φ(t) = [cos(t) sin(t); -sin(t) cos(t)]
        @test Q[1, 1] ≈ 0.0 atol = 1e-3   # cos(π/2)
        @test Q[1, 2] ≈ 1.0 atol = 1e-3   # sin(π/2)
        @test Q[2, 1] ≈ -1.0 atol = 1e-3  # -sin(π/2)
        @test Q[2, 2] ≈ 0.0 atol = 1e-3   # cos(π/2)
    end

    @testset "state_eov_rk4! - Type Stability" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test Float32
        x_f32 = Float32[1.0]
        Q_f32 = Matrix{Float32}(I, 1, 1)
        state_eov_rk4!(x_f32, Q_f32, linear_vel!, linear_jac!, 0.0f0, 1.0f0, 0.1f0)
        @test eltype(x_f32) == Float32
        @test eltype(Q_f32) == Float32

        # Test Float64
        x_f64 = Float64[1.0]
        Q_f64 = Matrix{Float64}(I, 1, 1)
        state_eov_rk4!(x_f64, Q_f64, linear_vel!, linear_jac!, 0.0, 1.0, 0.1)
        @test eltype(x_f64) == Float64
        @test eltype(Q_f64) == Float64
    end

    @testset "state_eov_rk4! - Long Integration" begin
        # Test integration over longer time span
        function linear_vel!(dx, x, t)
            dx[1] = -0.5 * x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -0.5
        end

        x = [10.0]
        Q = Matrix{Float64}(I, 1, 1)
        t0 = 0.0
        t1 = 5.0
        dt = 0.05

        state_eov_rk4!(x, Q, linear_vel!, linear_jac!, t0, t1, dt)

        x_analytical = 10.0 * exp(-0.5 * t1)
        Q_analytical = exp(-0.5 * t1)

        @test x[1] ≈ x_analytical atol = 1e-5
        @test Q[1, 1] ≈ Q_analytical atol = 1e-5
    end
end
