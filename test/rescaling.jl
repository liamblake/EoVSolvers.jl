using LinearAlgebra


@testset "Rescaling Integration Tests" begin
    @testset "state_eov_rk4_rescaling! - Basic Functionality" begin
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
        L = 10.0

        x_initial = copy(x)
        Q_initial = copy(Q)

        state_eov_rk4_rescaling!(x, Q, linear_vel!, linear_jac!, t0, t1, dt, L)

        # Analytical solutions
        x_analytical = x_initial[1] * exp(-t1)
        Q_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-6
        @test Q[1, 1] ≈ Q_analytical atol = 1e-6
    end

    @testset "state_eov_rk4_rescaling! - 2D System" begin
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
        L = 5.0

        state_eov_rk4_rescaling!(x, Q, linear2d_vel!, linear2d_jac!, t0, t1, dt, L)

        # Analytical solutions
        @test x[1] ≈ exp(-t1) atol = 1e-5
        @test x[2] ≈ 2 * exp(-2 * t1) atol = 1e-5

        # Q should be diagonal for this decoupled system
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-5
        @test Q[2, 2] ≈ exp(-2 * t1) atol = 1e-5
        @test abs(Q[1, 2]) < 1e-10
        @test abs(Q[2, 1]) < 1e-10
    end

    @testset "state_eov_rk4_rescaling! - Partial Final Step" begin
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
        L = 10.0

        state_eov_rk4_rescaling!(x, Q, linear_vel!, linear_jac!, t0, t1, dt, L)

        x_analytical = exp(-t1)
        Q_analytical = exp(-t1)

        @test x[1] ≈ x_analytical atol = 1e-6
        @test Q[1, 1] ≈ Q_analytical atol = 1e-6
    end

    @testset "state_eov_rk4_rescaling! - Preallocated Memory" begin
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
        L = 10.0

        # Preallocate memory
        state_alloc1 = similar(x)
        state_alloc2 = similar(x)
        state_alloc3 = similar(x)
        mat_alloc1 = similar(Q)
        mat_alloc2 = similar(Q)
        mat_alloc3 = similar(Q)

        state_eov_rk4_rescaling!(
            x,
            Q,
            linear_vel!,
            linear_jac!,
            t0,
            t1,
            dt,
            L;
            _state_alloc1 = state_alloc1,
            _state_alloc2 = state_alloc2,
            _state_alloc3 = state_alloc3,
            _mat_alloc1 = mat_alloc1,
            _mat_alloc2 = mat_alloc2,
            _mat_alloc3 = mat_alloc3,
        )

        @test x[1] ≈ exp(-t1) atol = 1e-6
        @test Q[1, 1] ≈ exp(-t1) atol = 1e-6
    end

    @testset "state_eov_rk4_rescaling! - Zero Integration Interval" begin
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
        L = 10.0

        state_eov_rk4_rescaling!(x, Q, linear_vel!, linear_jac!, t0, t1, dt, L)

        @test x[1] ≈ x_initial[1] atol = 1e-12
        @test Q[1, 1] ≈ Q_initial[1, 1] atol = 1e-12
    end

    @testset "state_eov_rk4_rescaling! - Harmonic Oscillator" begin
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
        L = 10.0

        state_eov_rk4_rescaling!(x, Q, harmonic_vel!, harmonic_jac!, t0, t1, dt, L)

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

    @testset "state_eov_rk4_rescaling! - Type Stability" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test Float32
        x_f32 = Float32[1.0]
        Q_f32 = Matrix{Float32}(I, 1, 1)
        state_eov_rk4_rescaling!(
            x_f32,
            Q_f32,
            linear_vel!,
            linear_jac!,
            0.0f0,
            1.0f0,
            0.1f0,
            10.0f0,
        )
        @test eltype(x_f32) == Float32
        @test eltype(Q_f32) == Float32

        # Test Float64
        x_f64 = Float64[1.0]
        Q_f64 = Matrix{Float64}(I, 1, 1)
        state_eov_rk4_rescaling!(x_f64, Q_f64, linear_vel!, linear_jac!, 0.0, 1.0, 0.1, 10.0)
        @test eltype(x_f64) == Float64
        @test eltype(Q_f64) == Float64
    end

    @testset "state_eov_rk4_rescaling! - Long Integration" begin
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
        L = 10.0

        state_eov_rk4_rescaling!(x, Q, linear_vel!, linear_jac!, t0, t1, dt, L)

        x_analytical = 10.0 * exp(-0.5 * t1)
        Q_analytical = exp(-0.5 * t1)

        @test x[1] ≈ x_analytical atol = 1e-5
        @test Q[1, 1] ≈ Q_analytical atol = 1e-5
    end

    @testset "state_eov_rk4_rescaling! - Different L Values" begin
        # Test that different L values produce the same result
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        t0 = 0.0
        t1 = 2.0
        dt = 0.1

        # Test with L = 1 (rescale every step)
        x_L1 = [1.0]
        Q_L1 = Matrix{Float64}(I, 1, 1)
        state_eov_rk4_rescaling!(x_L1, Q_L1, linear_vel!, linear_jac!, t0, t1, dt, 1.0)

        # Test with L = 5
        x_L5 = [1.0]
        Q_L5 = Matrix{Float64}(I, 1, 1)
        state_eov_rk4_rescaling!(x_L5, Q_L5, linear_vel!, linear_jac!, t0, t1, dt, 5.0)

        # Test with L = 100 (rarely rescale)
        x_L100 = [1.0]
        Q_L100 = Matrix{Float64}(I, 1, 1)
        state_eov_rk4_rescaling!(x_L100, Q_L100, linear_vel!, linear_jac!, t0, t1, dt, 100.0)

        x_analytical = exp(-t1)
        Q_analytical = exp(-t1)

        # All should give the same result regardless of L
        @test x_L1[1] ≈ x_analytical atol = 1e-6
        @test x_L5[1] ≈ x_analytical atol = 1e-6
        @test x_L100[1] ≈ x_analytical atol = 1e-6

        @test Q_L1[1, 1] ≈ Q_analytical atol = 1e-6
        @test Q_L5[1, 1] ≈ Q_analytical atol = 1e-6
        @test Q_L100[1, 1] ≈ Q_analytical atol = 1e-6
    end

    @testset "state_eov_rk4_rescaling! - Invalid L Threshold" begin
        # Test that L < 1 throws an error
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        x = [1.0]
        Q = Matrix{Float64}(I, 1, 1)

        @test_throws ArgumentError state_eov_rk4_rescaling!(
            x,
            Q,
            linear_vel!,
            linear_jac!,
            0.0,
            1.0,
            0.1,
            0.5,
        )
        @test_throws ArgumentError state_eov_rk4_rescaling!(
            x,
            Q,
            linear_vel!,
            linear_jac!,
            0.0,
            1.0,
            0.1,
            0.0,
        )
    end

    @testset "state_eov_rk4_rescaling! - Rescaling Stability for Growing System" begin
        # Test a system where state transition matrix grows exponentially
        # dx/dt = x (solution: x(t) = x₀ * exp(t))
        function growing_vel!(dx, x, t)
            dx[1] = x[1]
        end

        function growing_jac!(J, x, t)
            J[1, 1] = 1.0
        end

        x = [1.0]
        Q = Matrix{Float64}(I, 1, 1)
        t0 = 0.0
        t1 = 5.0  # STM grows to exp(5) ≈ 148
        dt = 0.01
        L = 10.0

        state_eov_rk4_rescaling!(x, Q, growing_vel!, growing_jac!, t0, t1, dt, L)

        x_analytical = exp(t1)
        Q_analytical = exp(t1)

        @test x[1] ≈ x_analytical atol = 1e-4
        @test Q[1, 1] ≈ Q_analytical atol = 1e-4
    end

    @testset "state_eov_rk4_rescaling! - Consistency with Non-Rescaling Method" begin
        # Compare with state_eov_rk4! for moderate-sized problems
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        t0 = 0.0
        t1 = 2.0
        dt = 0.05

        # Run with rescaling
        x_rescaling = [1.0]
        Q_rescaling = Matrix{Float64}(I, 1, 1)
        state_eov_rk4_rescaling!(
            x_rescaling,
            Q_rescaling,
            linear_vel!,
            linear_jac!,
            t0,
            t1,
            dt,
            100.0,
        )

        # Run without rescaling
        x_standard = [1.0]
        Q_standard = Matrix{Float64}(I, 1, 1)
        state_eov_rk4!(x_standard, Q_standard, linear_vel!, linear_jac!, t0, t1, dt)

        # Results should be very close
        @test x_rescaling[1] ≈ x_standard[1] atol = 1e-10
        @test Q_rescaling[1, 1] ≈ Q_standard[1, 1] atol = 1e-10
    end
end
