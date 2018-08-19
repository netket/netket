#include <chrono>

#ifdef WITH_BOOST_ODEINT
#include <boost/numeric/odeint.hpp>
#endif // WITH_BOOST_ODEINT

#include <Catch/catch.hpp>
#include <Eigen/Eigen>

#include "netket.hpp"
#include "Dynamics/time_evolution.hpp"
#include "Dynamics/TimeStepper/explicit_time_steppers.hpp"
#include "Dynamics/TimeStepper/controlled_time_steppers.hpp"

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
        using TimeStepper = ode::EulerTimeStepper<State>;
        TimeStepper stepper(dt, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, noop, initial, {t0, tmax, dt}, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
    SECTION("Euler with internal time step != dt")
    {
        using TimeStepper = ode::EulerTimeStepper<State>;
        TimeStepper stepper(0.3 * dt, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, noop, initial, {t0, tmax, dt}, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
    SECTION("RK4")
    {
        using TimeStepper = ode::RungeKutta4Stepper<State>;
        TimeStepper stepper(0.4 * dt, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, noop, initial, {t0, tmax, dt}, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
    SECTION("controlled Heun")
    {
        using TimeStepper = ode::HeunTimeStepper<State>;
        TimeStepper stepper(1e-6, 1e-6, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, noop, initial, {t0, tmax, dt}, observer);

        CHECK(ts.size() == 21);
        for(size_t j = 0; j < ts.size(); ++j)
        {
            double current_t = -10. + j;
            CHECK(Approx(ts[j]) == current_t);
        }
    }
    SECTION("controlled Dopri54")
    {
        using TimeStepper = ode::Dopri54TimeStepper<State>;
        TimeStepper stepper(1e-6, 1e-6, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, noop, initial, {t0, tmax, dt}, observer);

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
        using TimeStepper = ode::EulerTimeStepper<State>;
        TimeStepper stepper(dt, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, initial, {t0, tmax, dt}, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
    SECTION("Euler with internal time step != dt")
    {
        using TimeStepper = ode::EulerTimeStepper<State>;
        TimeStepper stepper(0.3 * dt, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, initial, {t0, tmax, dt}, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
    SECTION("RK4")
    {
        using TimeStepper = ode::RungeKutta4Stepper<State>;
        TimeStepper stepper(0.4 * dt, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, initial, {t0, tmax, dt}, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
    SECTION("controlled Heun")
    {
        using TimeStepper = ode::HeunTimeStepper<State>;
        TimeStepper stepper(1e-6, 1e-6, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, initial, {t0, tmax, dt}, observer);

        for(size_t j = 0; j < ts.size(); ++j)
        {
            CHECK(Approx(xs[j](0)).margin(m) == ts[j]);
        }
    }
    SECTION("controlled Dopri54")
    {
        using TimeStepper = ode::Dopri54TimeStepper<State>;
        TimeStepper stepper(1e-6, 1e-6, initial.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, initial, {t0, tmax, dt}, observer);

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
        using TimeStepper = ode::EulerTimeStepper<State>;
        TimeStepper stepper(5e-5 * dt, x.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, dt});

        const double m = 1e-3; // explicit Euler has a large error
        CHECK(Approx(x.norm()).margin(m) == 1.);
    }
    SECTION("RK4")
    {
        using TimeStepper = ode::RungeKutta4Stepper<State>;
        TimeStepper stepper(5e-4 * dt, x.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, dt});

        CHECK(Approx(x.norm()) == 1.);
    }
    SECTION("controlled Heun")
    {
        using TimeStepper = ode::HeunTimeStepper<State>;
        TimeStepper stepper(1e-6, 1e-6, x.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, dt});

        CHECK(Approx(x.norm()) == 1.);
    }
    SECTION("controlled Dopri54")
    {
        using TimeStepper = ode::Dopri54TimeStepper<State>;
        TimeStepper stepper(1e-9, 1e-9, x.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, dt});

        CHECK(Approx(x.norm()) == 1.);
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
    const auto gc = std::conj(g);

    const double delta = 1.;

    Eigen::MatrixXcd hamiltonian(3, 3);
    hamiltonian << -delta/2, g, g,
                    gc,      0, 0,
                    gc,      0, delta/2;

    auto eom = [&](const State& x, State& dxdt, double /*t*/) {
        dxdt.noalias() = -im * hamiltonian * x;
    };

    auto odeint_eom = [&](const OdeintState& x, OdeintState& dxdt, double /*t*/) {
        Eigen::Map<const State> x_map(x.data(), x.size());
        Eigen::Map<State> dxdt_map(dxdt.data(), dxdt.size());

        dxdt_map = -im * hamiltonian * x_map;
    };

    const double t0 = 0.;
    const double tmax = 100.;
    const double dt = 10.;

    State x(3);
    x << 1., 0., 0.;

    OdeintState xo{{1., 0.}, {0., 0.}, {0., 0.}};

    SECTION("explicit Euler")
    {
        using TimeStepper = ode::EulerTimeStepper<State>;
        TimeStepper stepper(5e-5 * dt, x.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, 5e-5 * dt});

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
        using TimeStepper = ode::RungeKutta4Stepper<State>;
        TimeStepper stepper(1e-4 * dt, x.size());
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, 1e-4 * dt});

        using OdeintStepper = odeint::runge_kutta4<OdeintState>;
        odeint::integrate_const(OdeintStepper(), odeint_eom, xo, t0, tmax, 1e-4 * dt);

        for(size_t i = 0; i < xo.size(); i++)
        {
            REQUIRE(Approx(x(i).real()) == xo[i].real());
            REQUIRE(Approx(x(i).imag()) == xo[i].imag());
        }
    }
    SECTION("controlled Dopri54")
    {
        using TimeStepper = ode::Dopri54TimeStepper<State>;
        TimeStepper stepper(1e-12, 1e-12, x.size());

        Stopwatch watch;
        ode::Integrate<TimeStepper, State>(stepper, eom, x, {t0, tmax, dt});
        auto elapsed = watch.elapsed();
        std::cout << "Dopri54 ellapsed time: " << elapsed.count() << "us" << std::endl;

        using OdeintStepper = odeint::runge_kutta_dopri5<OdeintState>;
        auto odeint_stepper = odeint::make_controlled<OdeintStepper>(1e-12, 1e-12);

        watch.restart();
        odeint::integrate_const(odeint_stepper, odeint_eom, xo, t0, tmax, dt);
        elapsed = watch.elapsed();
        std::cout << " odeint ellapsed time: " << elapsed.count() << "us" << std::endl;

        double m = 1e-17;
        for(size_t i = 0; i < xo.size(); i++)
        {
            REQUIRE(Approx(x(i).real()).margin(m) == xo[i].real());
            REQUIRE(Approx(x(i).imag()).margin(m) == xo[i].imag());
        }
    }
}
#endif

TEST_CASE("Time evolution driver produces sensible output", "[time-evolution]")
{
    json pars;

    pars["Hilbert"]["QuantumNumbers"] = {-1, 1};
    pars["Hilbert"]["Size"] = 1;

    json observable_json;
    observable_json["ActingOn"] = {{0}};
    observable_json["Operators"] = {{{-1, 2}, {2, 1}}};

    pars["Hamiltonian"] = observable_json;

    pars["TimeEvolution"] = {
            {"StepperType", "Dopri54"},
            {"OutputFiles", "test_output_%i.log"},
            {"AbsTol", 1e-9},
            {"RelTol", 1e-9},
            {"StartTime", 0.0},
            {"EndTime", 100.0},
            {"TimeStep", 1.0}
    };

    pars["TimeEvolution"]["InitialStates"] = {
        {{1, 0}, {0, 0}},
        {{0, 0}, {1, 0}},
    };

    {
        Stopwatch watch;
        RunTimeEvolution(pars);
        std::cout << "TimeEvolution elapsed: " << watch.elapsed().count() << "us" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    auto outfiles = std::vector<std::string>{
            "test_output_0.log",
            "test_output_1.log"};

    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    int mpisize;
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    for(size_t i = mpirank; i < outfiles.size(); i += mpisize)
    {
        auto fn = outfiles[i];
        INFO(fn);
        std::ifstream fin(fn);
        CHECK(fin.good());

        std::string line;
        int pos = 0;
        while(std::getline(fin, line))
        {
            INFO(line);
            json js = json::parse(line);

            REQUIRE(js.is_object());

            REQUIRE(js.find("Time") != js.end());
            double time = js["Time"].get<double>();
            CHECK(Approx(time) == pos);

            REQUIRE(js.find("State") != js.end());
            REQUIRE(js["State"].size() == 2); // States have Hilbert space dimension
            using StateType = std::vector<std::complex<double>>;
            auto state = js["State"].get<StateType>();

            pos++;
        }
    }
}
