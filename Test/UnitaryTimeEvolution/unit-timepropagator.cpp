#include <Catch/catch.hpp>
#include <Utils/all_utils.hpp>
#include <netket.hpp>
#include <boost/numeric/odeint.hpp>

#include "UnitaryTimeEvolution/TimeStepper/explicit_time_steppers.hpp"

using namespace netket;

TEST_CASE("Observer function is called once at each time step", "[time-evolution]")
{
    using State = Eigen::VectorXd;
    std::vector<double> ts;

    auto noop = [](const State& /*x*/, State& /*dxdt*/, double /*t*/) {};
    auto observer = [&](const State& /*x*/, double t) {
        ts.push_back(t);
    };

    const double t0 = -10.;
    const double tmax = 10.;
    const double dt = 1.;

    State initial(1);
    initial << 1.;

    SECTION("Euler with internal time step = dt")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(dt, initial.size());
        ode::Integrate<Stepper, State>(stepper, noop, initial, t0, tmax, dt, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
    SECTION("Euler with internal time step != dt")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(0.3 * dt, initial.size());
        ode::Integrate<Stepper, State>(stepper, noop, initial, t0, tmax, dt, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
    SECTION("RK4")
    {
        using Stepper = ode::RungeKutta4Stepper<State>;
        Stepper stepper(0.4 * dt, initial.size());
        ode::Integrate<Stepper, State>(stepper, noop, initial, t0, tmax, dt, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
}

TEST_CASE("Integrators can propagate trivial ODE", "[time-evolution]")
{
    using State = Eigen::VectorXd;

    std::vector<double> ts;
    std::vector<State> xs;

    auto eom = [](const State& /*x*/, State& dxdt, double /*t*/) {
        dxdt(0) = 1.0;
    };
    auto observer = [&](const State& x, double t) {
        xs.push_back(x);
        ts.push_back(t);
    };

    const double t0 = -10.;
    const double tmax = 10.;
    const double dt = 1.;

    State initial(1);
    initial << t0;

    // use small absolute margin
    const double m = 1e-14;
    SECTION("Euler with internal time step = dt")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(dt, initial.size());
        ode::Integrate<Stepper, State>(stepper, eom, initial, t0, tmax, dt, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
    SECTION("Euler with internal time step != dt")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(0.3 * dt, initial.size());
        ode::Integrate<Stepper, State>(stepper, eom, initial, t0, tmax, dt, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
    SECTION("RK4")
    {
        using Stepper = ode::RungeKutta4Stepper<State>;
        Stepper stepper(0.4 * dt, initial.size());
        ode::Integrate<Stepper, State>(stepper, eom, initial, t0, tmax, dt, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
}

TEST_CASE("Integrators approximately conserve norm when propagating Schroedinger eq", "[time-evolution]")
{
    using State = Eigen::VectorXcd;

    const double sqrt2 = std::sqrt(2.);
    const std::complex<double> im(0, 1);

    const std::complex<double> g(1/sqrt2, 1/sqrt2);

    const double delta = 1.;

    Eigen::MatrixXcd hamiltonian(2, 2);
    hamiltonian << -delta/2, g, std::conj(g), delta/2;

    auto eom = [&](const State& x, State& dxdt, double /*t*/) {
        dxdt.noalias() = -im * hamiltonian * x;
    };

    const double t0 = 0.;
    const double tmax = 20.;
    const double dt = 1.;

    State x(2);
    x << 1., 0;

    SECTION("explicit Euler")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(5e-5 * dt, x.size());
        ode::Integrate<Stepper, State>(stepper, eom, x, t0, tmax, dt);

        const double m = 1e-3; // explicit Euler has a large error
        CHECK(Approx(x.norm()).margin(m) == 1.);
    }
    SECTION("RK4")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(5e-4 * dt, x.size());
        ode::Integrate<Stepper, State>(stepper, eom, x, t0, tmax, dt);

        const double m = 1e-2;
        CHECK(Approx(x.norm()).margin(m) == 1.);
    }
}

#ifdef WITH_BOOST_ODEINT

namespace odeint = boost::numeric::odeint;

TEST_CASE("Comparison with boost::odeint for Schroedinger eq", "[time-evolution]")
{
    using State = Eigen::VectorXcd;
    using OdeintState = std::vector<std::complex<double>>;

    const double sqrt2 = std::sqrt(2.);
    const std::complex<double> im(0, 1);

    const std::complex<double> g(1/sqrt2, 1/sqrt2);

    const double delta = 1.;

    Eigen::MatrixXcd hamiltonian(2, 2);
    hamiltonian << -delta/2, g, std::conj(g), delta/2;

    auto eom = [&](const State& x, State& dxdt, double /*t*/) {
        dxdt.noalias() = -im * hamiltonian * x;
    };

    auto odeint_eom = [&](const OdeintState& x, OdeintState& dxdt, double /*t*/) {
        Eigen::Map<const State> x_map(x.data(), x.size());
        Eigen::Map<State> dxdt_map(dxdt.data(), dxdt.size());

        dxdt_map = -im * hamiltonian * x_map;
    };

    const double t0 = 0.;
    const double tmax = 10.;
    const double dt = 1.;

    State x(2);
    x << 1., 0.;

    OdeintState xo{{1., 0.}, {0., 0.}};

    SECTION("explicit Euler")
    {
        using Stepper = ode::EulerTimeStepper<State>;
        Stepper stepper(5e-5 * dt, x.size());
        ode::Integrate<Stepper, State>(stepper, eom, x, t0, tmax, 5e-5 * dt);

        using OdeintStepper = odeint::euler<OdeintState>;
        odeint::integrate_const(OdeintStepper(), odeint_eom, xo, t0, tmax, 5e-5 * dt);

        for(size_t i = 0; i < xo.size(); i++)
        {
            REQUIRE(Approx(x(i).real()) == xo[i].real());
            REQUIRE(Approx(x(i).imag()) == xo[i].imag());
        }
    }
    SECTION("RK4")
    {
        using Stepper = ode::RungeKutta4Stepper<State>;
        Stepper stepper(1e-4 * dt, x.size());
        ode::Integrate<Stepper, State>(stepper, eom, x, t0, tmax, 1e-4 * dt);

        using OdeintStepper = odeint::runge_kutta4<OdeintState>;
        odeint::integrate_const(OdeintStepper(), odeint_eom, xo, t0, tmax, 1e-4 * dt);

        for(size_t i = 0; i < xo.size(); i++)
        {
            REQUIRE(Approx(x(i).real()) == xo[i].real());
            REQUIRE(Approx(x(i).imag()) == xo[i].imag());
        }
    }
}
#endif
