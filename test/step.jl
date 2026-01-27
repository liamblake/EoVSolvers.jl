using LinearAlgebra

@testset "Solver Tests" begin
    @testset "rk4_step! - Basic Functionality" begin
        # Test simple linear ODE: dx/dt = -x
        # Analytical solution: x(t) = x₀ * exp(-t)
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test single step
        x = [1.0]
        dt = 0.1
        t0 = 0.0

        x_initial = copy(x)
        rk4_step!(x, linear_vel!, t0, dt)

        # Analytical solution at t = 0.1
        x_analytical = x_initial[1] * exp(-dt)

        @test length(x) == 1
        @test x[1] ≈ x_analytical atol = 1e-6
        @test x !== x_initial  # Should modify in-place
    end

    @testset "rk4_step! - 2D System" begin
        # Test 2D harmonic oscillator: d²x/dt² = -x
        # State: [x, dx/dt], dynamics: [dx/dt, -x]
        function harmonic_vel!(dx, x, t)
            dx[1] = x[2]      # dx/dt = velocity
            dx[2] = -x[1]     # d²x/dt² = -x
        end

        function harmonic_jac!(J, x, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -1.0
            J[2, 2] = 0.0
        end

        # Initial condition: x=1, v=0
        x = [1.0, 0.0]
        dt = 0.1
        t0 = 0.0

        rk4_step!(x, harmonic_vel!, t0, dt)

        # Analytical solution: x(t) = cos(t), v(t) = -sin(t)
        x_analytical = [cos(dt), -sin(dt)]

        @test x[1] ≈ x_analytical[1] atol = 1e-4
        @test x[2] ≈ x_analytical[2] atol = 1e-4
    end

    @testset "rk4_step! - Order of Accuracy" begin
        # Test convergence rate for dx/dt = -x
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test different step sizes
        dts = [0.1, 0.05, 0.025, 0.0125]
        errors = Float64[]

        for dt in dts
            x = [1.0]
            rk4_step!(x, linear_vel!, 0.0, dt)

            # Analytical solution
            x_exact = exp(-dt)
            error = abs(x[1] - x_exact)
            push!(errors, error)
        end

        # Check 4th order convergence: error ~ dt^4
        for i = 2:length(errors)
            ratio = errors[i-1] / errors[i]
            expected_ratio = (dts[i-1] / dts[i])^4
            @test_broken ratio ≈ expected_ratio rtol = 0.1  # Allow 10% tolerance for numerical effects
        end
    end

    @testset "rk4_step! - Memory Safety" begin
        function simple_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = x[1] - x[2]
        end

        function simple_jac!(J, x, t)
            J[1, 1] = -1.0
            J[1, 2] = 0.0
            J[2, 1] = 1.0
            J[2, 2] = -1.0
        end

        # Test that original array is modified in-place
        x_original = [1.0, 2.0]
        x_test = copy(x_original)

        rk4_step!(x_test, simple_vel!, 0.0, 0.1)

        @test x_test !== x_original  # Different objects
        @test x_test != x_original   # Different values
        @test length(x_test) == length(x_original)

        # Test that model storage arrays are not aliased
        x1 = [1.0, 2.0]
        x2 = [3.0, 4.0]

        rk4_step!(x1, simple_vel!, 0.0, 0.1)
        rk4_step!(x2, simple_vel!, 0.0, 0.1)

        # Results should be independent
        @test x1[1] ≈ exp(-0.1) atol = 1e-6
        @test x2[1] ≈ 3 * exp(-0.1) atol = 1e-6
    end

    @testset "rk4_step! - Edge Cases" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test zero step size
        x = [1.0]
        x_initial = copy(x)
        rk4_step!(x, linear_vel!, 0.0, 0.0)
        @test x[1] ≈ x_initial[1] atol = 1e-12

        # Test negative step size (backward integration)
        x = [1.0]
        rk4_step!(x, linear_vel!, 0.0, -0.1)
        x_expected = exp(0.1)  # Backward integration
        @test x[1] ≈ x_expected atol = 1e-6

        # Test with zero initial condition
        x = [0.0]
        rk4_step!(x, linear_vel!, 0.0, 0.1)
        @test x[1] ≈ 0.0 atol = 1e-12
    end

    @testset "rk4_step! - Time-Dependent System" begin
        # Test dx/dt = t (solution: x(t) = x₀ + t²/2 - t₀²/2)
        function time_dep_vel!(dx, x, t)
            dx[1] = t
        end

        function time_dep_jac!(J, x, t)
            J[1, 1] = 0.0
        end

        x = [1.0]
        t0 = 0.5
        dt = 0.2

        rk4_step!(x, time_dep_vel!, t0, dt)

        # Analytical solution: x(t) = 1 + (t²/2 - t₀²/2)
        t1 = t0 + dt
        x_analytical = 1.0 + (t1^2 / 2 - t0^2 / 2)

        @test x[1] ≈ x_analytical atol = 1e-6
    end

    @testset "rk4_step! - Type Stability" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test Float32
        x_f32 = Float32[1.0]
        rk4_step!(x_f32, linear_vel!, 0.0f0, 0.1f0)
        @test eltype(x_f32) == Float32

        # Test Float64
        x_f64 = Float64[1.0]
        rk4_step!(x_f64, linear_vel!, 0.0, 0.1)
        @test eltype(x_f64) == Float64

        # Test BigFloat (if needed for high precision)
        x_big = BigFloat[1.0]
        rk4_step!(x_big, linear_vel!, BigFloat(0.0), BigFloat(0.1))
        @test eltype(x_big) == BigFloat
    end

    @testset "rk4_step! - Multiple Steps Integration" begin
        # Test that multiple RK4 steps approximate analytical solution
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Integrate from t=0 to t=1 with multiple steps
        x = [1.0]
        t = 0.0
        dt = 0.01
        n_steps = 100

        for i = 1:n_steps
            rk4_step!(x, linear_vel!, t, dt)
            t += dt
        end

        # Compare with analytical solution
        x_analytical = exp(-1.0)
        @test x[1] ≈ x_analytical atol = 1e-8
    end

    @testset "rk4_step! - Nonlinear System" begin
        # Test Van der Pol oscillator (simplified): dx/dt = y, dy/dt = -x
        function vdp_vel!(dx, x, t)
            dx[1] = x[2]
            dx[2] = -x[1]
        end

        function vdp_jac!(J, x, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -1.0
            J[2, 2] = 0.0
        end

        # Initial condition and step
        x = [1.0, 0.0]
        dt = 0.01

        # Take several steps and check energy conservation
        # For harmonic oscillator: E = 0.5 * (x² + y²) should be conserved
        initial_energy = 0.5 * (x[1]^2 + x[2]^2)

        for i = 1:100
            rk4_step!(x, vdp_vel!, (i - 1) * dt, dt)
        end

        final_energy = 0.5 * (x[1]^2 + x[2]^2)

        # Energy should be approximately conserved (RK4 has small energy drift)
        @test abs(final_energy - initial_energy) < 0.01
    end

    @testset "rk4_step_state_eov! - Basic Functionality" begin
        # Test simple linear ODE: dx/dt = -x with Jacobian evolution
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Initial state and identity Jacobian
        x = [1.0]
        J = Matrix{Float64}(I, 1, 1)  # Start with identity
        dt = 0.1
        t0 = 0.0

        x_initial = copy(x)
        J_initial = copy(J)

        rk4_step_state_eov!(x, J, linear_vel!, linear_jac!, t0, dt)

        # Analytical solution: x(t) = x₀ * exp(-t), J(t) = exp(-t)
        x_analytical = x_initial[1] * exp(-dt)
        J_analytical = exp(-dt)

        @test x[1] ≈ x_analytical atol = 1e-6
        @test J[1, 1] ≈ J_analytical atol = 1e-6
        @test x !== x_initial && J !== J_initial  # Should modify in-place
    end

    @testset "rk4_step_state_eov! - 2D Linear System" begin
        # Test 2D linear system: dx/dt = Ax where A = [-1 0; 1 -2]
        function linear2d_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = x[1] - 2 * x[2]
        end

        function linear2d_jac!(J, x, t)
            J[1, 1] = -1.0
            J[1, 2] = 0.0
            J[2, 1] = 1.0
            J[2, 2] = -2.0
        end

        x = [1.0, 0.5]
        J = Matrix{Float64}(I, 2, 2)
        dt = 0.1

        rk4_step_state_eov!(x, J, linear2d_vel!, linear2d_jac!, 0.0, dt)

        # For linear system dx/dt = Ax, the fundamental matrix is Φ(t) = exp(At)
        A = [-1.0 0.0; 1.0 -2.0]
        Φ_analytical = exp(A * dt)
        x_analytical = Φ_analytical * [1.0, 0.5]

        @test x[1] ≈ x_analytical[1] atol = 1e-4
        @test x[2] ≈ x_analytical[2] atol = 1e-4
        @test J[1, 1] ≈ Φ_analytical[1, 1] atol = 1e-4
        @test J[1, 2] ≈ Φ_analytical[1, 2] atol = 1e-4
        @test J[2, 1] ≈ Φ_analytical[2, 1] atol = 1e-4
        @test J[2, 2] ≈ Φ_analytical[2, 2] atol = 1e-4
    end

    @testset "rk4_step_state_eov! - Nonlinear Pendulum" begin
        # Test nonlinear pendulum: θ̈ + sin(θ) = 0
        # State: [θ, θ̇], dynamics: [θ̇, -sin(θ)]
        function pendulum_vel!(dx, x, t)
            dx[1] = x[2]        # θ̇
            dx[2] = -sin(x[1])  # -sin(θ)
        end

        function pendulum_jac!(J, x, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -cos(x[1])  # ∂(-sin(θ))/∂θ = -cos(θ)
            J[2, 2] = 0.0
        end

        # Small angle - linearization should be close to harmonic oscillator
        x = [0.1, 0.0]  # θ=0.1, θ̇=0
        J = Matrix{Float64}(I, 2, 2)
        dt = 0.05

        rk4_step_state_eov!(x, J, pendulum_vel!, pendulum_jac!, 0.0, dt)

        # For small angles, sin(θ) ≈ θ, so system becomes harmonic oscillator
        # Check that Jacobian evolution makes sense
        @test J[1, 1] ≈ 1.0 atol = 0.1     # Should be close to 1 for small time
        @test J[1, 2] > 0                   # ∂θ/∂θ̇₀ should be positive
        @test J[2, 1] < 0                   # ∂θ̇/∂θ₀ should be negative (restoring force)
        @test abs(J[2, 2] - 1.0) < 0.1     # ∂θ̇/∂θ̇₀ should be close to 1
    end

    @testset "rk4_step_state_eov! - Consistency with Separate Integration" begin
        # Verify that combined integration matches separate state and linearized integration
        function nl_vel!(dx, x, t)
            dx[1] = x[2]
            dx[2] = -x[1] - 0.1 * x[1]^3  # Nonlinear spring
        end

        function nl_jac!(J, x, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -1.0 - 0.3 * x[1]^2  # ∂(-x - 0.1*x³)/∂x = -1 - 0.3*x²
            J[2, 2] = 0.0
        end

        # Test point
        x0 = [0.5, 0.2]
        J0 = Matrix{Float64}(I, 2, 2)
        dt = 0.01

        # Combined integration
        x_combined = copy(x0)
        J_combined = copy(J0)
        rk4_step_state_eov!(x_combined, J_combined, nl_vel!, nl_jac!, 0.0, dt)

        # Separate integration - state only
        x_separate = copy(x0)
        rk4_step!(x_separate, nl_vel!, 0.0, dt)

        # States should match
        @test x_combined[1] ≈ x_separate[1] atol = 1e-10
        @test x_combined[2] ≈ x_separate[2] atol = 1e-10

        # Jacobian should be reasonable (positive determinant for small time)
        @test det(J_combined) > 0
    end

    @testset "rk4_step_state_eov! - Memory Safety and Independence" begin
        function simple_vel!(dx, x, t)
            dx[1] = -x[1]
            dx[2] = x[1] - x[2]
        end

        function simple_jac!(J, x, t)
            J[1, 1] = -1.0
            J[1, 2] = 0.0
            J[2, 1] = 1.0
            J[2, 2] = -1.0
        end

        # Test independence of multiple calls
        x1 = [1.0, 2.0]
        J1 = [1.0 0.5; -0.5 1.0]  # Non-identity initial Jacobian
        x1_orig = copy(x1)
        J1_orig = copy(J1)

        x2 = [3.0, 4.0]
        J2 = [2.0 1.0; 0.0 2.0]
        x2_orig = copy(x2)
        J2_orig = copy(J2)

        # Integrate both
        rk4_step_state_eov!(x1, J1, simple_vel!, simple_jac!, 0.0, 0.1)
        rk4_step_state_eov!(x2, J2, simple_vel!, simple_jac!, 0.0, 0.1)

        # Results should be independent and different from originals
        @test x1 != x1_orig && J1 != J1_orig
        @test x2 != x2_orig && J2 != J2_orig
        @test x1 != x2  # Different results for different inputs
    end

    @testset "rk4_step_state_eov! - Time-Dependent System" begin
        # Test system with time-dependent Jacobian
        function time_vel!(dx, x, t)
            dx[1] = t * x[1]  # dx/dt = tx
        end

        function time_jac!(J, x, t)
            J[1, 1] = t  # ∂(tx)/∂x = t
        end

        x = [1.0]
        J = reshape([1.0], 1, 1)
        t0 = 0.5
        dt = 0.1

        rk4_step_state_eov!(x, J, time_vel!, time_jac!, t0, dt)

        # Analytical: x(t) = x₀ * exp(t²/2 - t₀²/2)
        # Jacobian: ∂x/∂x₀ = exp(t²/2 - t₀²/2)
        t1 = t0 + dt
        factor = exp((t1^2 - t0^2) / 2)
        x_analytical = 1.0 * factor
        J_analytical = factor

        @test x[1] ≈ x_analytical atol = 1e-6
        @test J[1, 1] ≈ J_analytical atol = 1e-6
    end

    @testset "rk4_step_state_eov! - Conservation Properties" begin
        # Test Hamiltonian system with Jacobian properties
        function hamiltonian_vel!(dx, x, t)
            dx[1] = x[2]   # q̇ = p
            dx[2] = -x[1]  # ṗ = -q (harmonic oscillator)
        end

        function hamiltonian_jac!(J, x, t)
            J[1, 1] = 0.0
            J[1, 2] = 1.0
            J[2, 1] = -1.0
            J[2, 2] = 0.0
        end

        x = [1.0, 0.0]
        J = Matrix{Float64}(I, 2, 2)
        dt = 0.01

        # Take several steps
        for i = 1:100
            rk4_step_state_eov!(x, J, hamiltonian_vel!, hamiltonian_jac!, (i - 1) * dt, dt)
        end

        # For Hamiltonian systems, Jacobian determinant should be preserved
        @test abs(det(J) - 1.0) < 0.05  # Allow small numerical drift

        # Energy should be approximately conserved
        energy = 0.5 * (x[1]^2 + x[2]^2)
        @test abs(energy - 0.5) < 0.01
    end

    @testset "rk4_step_state_eov! - Type Stability" begin
        function simple_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function simple_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test Float32
        x_f32 = Float32[1.0]
        J_f32 = Matrix{Float32}(I, 1, 1)
        rk4_step_state_eov!(x_f32, J_f32, simple_vel!, simple_jac!, 0.0f0, 0.1f0)
        @test eltype(x_f32) == Float32
        @test eltype(J_f32) == Float32

        # Test Float64
        x_f64 = Float64[1.0]
        J_f64 = Matrix{Float64}(I, 1, 1)
        rk4_step_state_eov!(x_f64, J_f64, simple_vel!, simple_jac!, 0.0, 0.1)
        @test eltype(x_f64) == Float64
        @test eltype(J_f64) == Float64
    end

    @testset "rk4_step_state_eov! - Zero and Edge Cases" begin
        function linear_vel!(dx, x, t)
            dx[1] = -x[1]
        end

        function linear_jac!(J, x, t)
            J[1, 1] = -1.0
        end

        # Test zero step size
        x = [1.0]
        J = reshape([2.0], 1, 1)  # Non-identity initial
        x_initial = copy(x)
        J_initial = copy(J)
        rk4_step_state_eov!(x, J, linear_vel!, linear_jac!, 0.0, 0.0)
        @test x[1] ≈ x_initial[1] atol = 1e-12
        @test J[1, 1] ≈ J_initial[1, 1] atol = 1e-12

        # Test with zero initial state
        x = [0.0]
        J = reshape([1.0], 1, 1)
        rk4_step_state_eov!(x, J, linear_vel!, linear_jac!, 0.0, 0.1)
        @test x[1] ≈ 0.0 atol = 1e-12
        @test J[1, 1] ≈ exp(-0.1) atol = 1e-6  # Jacobian evolution independent of state for linear system

        # Test negative step size
        x = [1.0]
        J = Matrix{Float64}(I, 1, 1)
        rk4_step_state_eov!(x, J, linear_vel!, linear_jac!, 0.0, -0.1)
        x_expected = exp(0.1)  # Backward integration
        J_expected = exp(0.1)
        @test x[1] ≈ x_expected atol = 1e-6
        @test J[1, 1] ≈ J_expected atol = 1e-6
    end
end
