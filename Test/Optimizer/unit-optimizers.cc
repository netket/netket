// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include "catch.hpp"
#include "netket.hpp"

#include "optimizer_input_tests.hpp"
const Complex im(0, 1);
// const Complex I(0.0, 1.0);
const double pi = 3.14159265358979323846;

// Check that optimizer steps correctly. This means parameters initialized
// correctly, real and imag parts not getting mixed, etc.
TEST_CASE("optimizers step twice correctly", "[optimizer]") {
  auto input_tests = GetOptimizerInputs();
  std::cout << "# Correct output tests size = " << 7 << std::endl;

  for (std::size_t it = 0; it < 7; it++) {
    std::string name = input_tests[it].dump();

    if (input_tests[it]["Optimizer"]["Name"] == "Sgd") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd sol(2);
        sol << 4.9 - 0.4 * im, 4.8 - 0.5 * im;
        Eigen::VectorXcd sol2(2);
        sol2 << 4.8 - 0.8 * im, 4.6 - 1.0 * im;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-7);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-7);  // update twice correctly
      }
    }

    else if (input_tests[it]["Optimizer"]["Name"] == "AdaDelta") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        double eps = input_tests[it]["Optimizer"]["Epscut"];
        double rho = input_tests[it]["Optimizer"]["Rho"];

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd diff(2);
        diff << -grad(0).real() *
                        std::sqrt(eps / (grad(0).real() * (1. - rho) + eps)) -
                    grad(0).imag() *
                        std::sqrt(
                            eps /
                            (std::pow(grad(0).imag(), 2) * (1. - rho) + eps)) *
                        im,
            -grad(1).real() *
                    std::sqrt(eps / (std::pow(grad(1).real(), 2) * (1. - rho) +
                                     eps)) -
                grad(1).imag() *
                    std::sqrt(eps / (std::pow(grad(1).imag(), 2) * (1. - rho) +
                                     eps)) *
                    im;
        Eigen::VectorXcd sol(2);
        sol << params + diff;
        Eigen::VectorXcd diff2(2);
        diff2 << -grad(0).real() *
                         std::sqrt(
                             ((1. - rho) * std::pow(diff(0).real(), 2) + eps) /
                             ((1. + rho) * (1. - rho) *
                                  std::pow(grad(0).real(), 2) +
                              eps)) -
                     grad(0).imag() *
                         std::sqrt(
                             ((1. - rho) * std::pow(diff(0).imag(), 2) + eps) /
                             ((1. + rho) * (1. - rho) *
                                  std::pow(grad(0).imag(), 2) +
                              eps)) *
                         im,
            -grad(1).real() *
                    std::sqrt(
                        ((1. - rho) * std::pow(diff(1).real(), 2) + eps) /
                        ((1. + rho) * (1. - rho) * std::pow(grad(1).real(), 2) +
                         eps)) -
                grad(1).imag() *
                    std::sqrt(
                        ((1. - rho) * std::pow(diff(1).imag(), 2) + eps) /
                        ((1. + rho) * (1. - rho) * std::pow(grad(1).imag(), 2) +
                         eps)) *
                    im;
        Eigen::VectorXcd sol2(2);
        sol2 << params + diff + diff2;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-7);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-7);  // update twice correctly*/
      }
    }

    else if (input_tests[it]["Optimizer"]["Name"] == "AdaGrad") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        double eta = input_tests[it]["Optimizer"]["LearningRate"];
        double eps = input_tests[it]["Optimizer"]["Epscut"];

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd diff(2);
        diff << -eta * grad(0).real() /
                        std::sqrt(std::pow(grad(0).real(), 2) + eps) -
                    eta * grad(0).imag() /
                        std::sqrt(std::pow(grad(0).imag(), 2) + eps) * im,
            -eta * grad(1).real() /
                    std::sqrt(std::pow(grad(1).real(), 2) + eps) -
                eta * grad(1).imag() /
                    std::sqrt(std::pow(grad(1).imag(), 2) + eps) * im;
        Eigen::VectorXcd sol(2);
        sol << params + diff;
        Eigen::VectorXcd diff2(2);
        diff2 << -eta * grad(0).real() /
                         std::sqrt(2. * std::pow(grad(0).real(), 2) + eps) -
                     eta * grad(0).imag() /
                         std::sqrt(2. * std::pow(grad(0).imag(), 2) + eps) * im,
            -eta * grad(1).real() /
                    std::sqrt(2. * std::pow(grad(1).real(), 2) + eps) -
                eta * grad(1).imag() /
                    std::sqrt(2. * std::pow(grad(1).imag(), 2) + eps) * im;
        Eigen::VectorXcd sol2(2);
        sol2 << params + diff + diff2;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-7);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-7);  // update twice correctly
      }
    }

    else if (input_tests[it]["Optimizer"]["Name"] == "AdaMax") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        double alpha = input_tests[it]["Optimizer"]["Alpha"];

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd diff(2);
        diff << -alpha - alpha * im, -alpha - alpha * im;
        Eigen::VectorXcd sol(2);
        sol << params + diff;
        Eigen::VectorXcd sol2(2);
        sol2 << params + 2 * diff;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-7);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-7);  // update twice correctly
      }
    }

    else if (input_tests[it]["Optimizer"]["Name"] == "AMSGrad") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        double eta = input_tests[it]["Optimizer"]["LearningRate"];
        double beta1 = input_tests[it]["Optimizer"]["Beta1"];
        double beta2 = input_tests[it]["Optimizer"]["Beta2"];
        double eps = input_tests[it]["Optimizer"]["Epscut"];

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd diff(2);
        diff << -eta * (1 - beta1) * grad(0).real() /
                        std::sqrt((1 - beta2) * std::pow(grad(0).real(), 2) +
                                  eps) -
                    eta * (1 - beta1) * grad(0).imag() /
                        std::sqrt((1 - beta2) * std::pow(grad(0).imag(), 2) +
                                  eps) *
                        im,
            -eta * (1 - beta1) * grad(1).real() /
                    std::sqrt((1 - beta2) * std::pow(grad(1).real(), 2) + eps) -
                eta * (1 - beta1) * grad(1).imag() /
                    std::sqrt((1 - beta2) * std::pow(grad(1).imag(), 2) + eps) *
                    im;
        Eigen::VectorXcd sol(2);
        sol << params + diff;
        Eigen::VectorXcd diff2(2);
        diff2 << -eta * (1 + beta1) * (1 - beta1) * grad(0).real() /
                         std::sqrt((1 - beta2) * (1 + beta2) *
                                       std::pow(grad(0).real(), 2) +
                                   eps) -
                     eta * (1 + beta1) * (1 - beta1) * grad(0).imag() /
                         std::sqrt((1 - beta2) * (1 + beta2) *
                                       std::pow(grad(0).imag(), 2) +
                                   eps) *
                         im,
            -eta * (1 + beta1) * (1 - beta1) * grad(1).real() /
                    std::sqrt((1 - beta2) * (1 + beta2) *
                                  std::pow(grad(1).real(), 2) +
                              eps) -
                eta * (1 + beta1) * (1 - beta1) * grad(1).imag() /
                    std::sqrt((1 - beta2) * (1 + beta2) *
                                  std::pow(grad(1).imag(), 2) +
                              eps) *
                    im;
        Eigen::VectorXcd sol2(2);
        sol2 << params + diff + diff2;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-5);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-5);  // update twice correctly
      }
    }

    else if (input_tests[it]["Optimizer"]["Name"] == "RMSProp") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        double eta = input_tests[it]["Optimizer"]["LearningRate"];
        double beta = input_tests[it]["Optimizer"]["Beta"];
        double eps = input_tests[it]["Optimizer"]["Epscut"];

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd diff(2);
        diff << -eta * grad(0).real() /
                        std::sqrt((1 - beta) * std::pow(grad(0).real(), 2) +
                                  eps) -
                    eta * grad(0).imag() /
                        std::sqrt((1 - beta) * std::pow(grad(0).imag(), 2) +
                                  eps) *
                        im,
            -eta * grad(1).real() /
                    std::sqrt((1 - beta) * std::pow(grad(1).real(), 2) + eps) -
                eta * grad(1).imag() /
                    std::sqrt((1 - beta) * std::pow(grad(1).imag(), 2) + eps) *
                    im;
        Eigen::VectorXcd sol(2);
        sol << params + diff;
        Eigen::VectorXcd diff2(2);
        diff2 << -eta * grad(0).real() /
                         std::sqrt((1 + beta) * (1 - beta) *
                                       std::pow(grad(0).real(), 2) +
                                   eps) -
                     eta * grad(0).imag() /
                         std::sqrt((1 + beta) * (1 - beta) *
                                       std::pow(grad(0).imag(), 2) +
                                   eps) *
                         im,
            -eta * grad(1).real() /
                    std::sqrt((1 + beta) * (1 - beta) *
                                  std::pow(grad(1).real(), 2) +
                              eps) -
                eta * grad(1).imag() /
                    std::sqrt((1 + beta) * (1 - beta) *
                                  std::pow(grad(1).imag(), 2) +
                              eps) *
                    im;
        Eigen::VectorXcd sol2(2);
        sol2 << params + diff + diff2;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-7);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-7);  // update twice correctly
      }
    } else if (input_tests[it]["Optimizer"]["Name"] == "Momentum") {
      SECTION("Stepper test (" + std::to_string(it) + ") on " + name) {
        netket::Optimizer optimizer(input_tests[it]);

        double eta = input_tests[it]["Optimizer"]["LearningRate"];
        double beta = input_tests[it]["Optimizer"]["Beta"];

        Eigen::VectorXcd grad(2);
        grad << 1.0 + 4.0 * im, 2.0 + 5.0 * im;
        Eigen::VectorXcd params(2);
        params << 5.0, 5.0;
        Eigen::VectorXcd sol(2);
        sol << params - eta * (1. - beta) * grad;
        Eigen::VectorXcd sol2(2);
        sol2 << params - eta * (1. - beta) * (2. + beta) * grad;

        optimizer.Init(params);
        optimizer.Update(grad, params);
        REQUIRE((params - sol).norm() < 1.0e-7);  // update once correctly
        optimizer.Update(grad, params);
        REQUIRE((params - sol2).norm() < 1.0e-7);  // update twice correctly
      }
    }
  }
}

TEST_CASE("optimizers correctly minimize Matyas function", "[optimizer]") {
  auto input_tests = GetOptimizerInputs();
  std::cout << "# Matyas input tests size = " << 7 << std::endl;

  for (std::size_t it = 0; it < 7; it++) {
    std::string name = input_tests[it].dump();

    SECTION("Optimizer test (" + std::to_string(it) + ") on " + name) {
      float err = 1.0e8;
      float tol = 0.1;
      int iter = 0;

      const double mean = 0.0;
      const double stddev = 0.1;
      std::default_random_engine generator;
      std::normal_distribution<double> dist(mean, stddev);

      Eigen::VectorXd sol(2);
      sol << 0, 0;
      Eigen::VectorXd grad(2);
      Eigen::VectorXd params(2);
      params << 5.0, 5.0;

      netket::Optimizer optimizer(input_tests[it]);
      optimizer.Init(params);

      while (err > tol and iter < 5e4) {
        grad(0) = 0.52 * params(0) - 0.48 * params(1) + dist(generator);
        grad(1) = 0.52 * params(1) - 0.48 * params(0) + dist(generator);

        optimizer.Update(grad, params);
        err = (params - sol).norm();
        iter += 1;
      }

      REQUIRE(err <= tol);
    }
  }
}

TEST_CASE("optimizers correctly minimize Beale function", "[optimizer]") {
  auto input_tests = GetOptimizerInputs();
  std::cout << "# Beale input tests size = " << 7 << std::endl;

  for (std::size_t it = 7; it < 14; it++) {
    std::string name = input_tests[it].dump();

    SECTION("Optimizer test (" + std::to_string(it) + ") on " + name) {
      float err = 1.0e8;
      float tol = 0.1;
      int iter = 0;

      const double mean = 0.0;
      const double stddev = 0.1;
      std::default_random_engine generator;
      std::normal_distribution<double> dist(mean, stddev);

      Eigen::VectorXd sol(2);
      sol << 3, 0.5;
      Eigen::VectorXd grad(2);
      Eigen::VectorXd params(2);
      params << 4.0, 3.0;

      netket::Optimizer optimizer(input_tests[it]);
      optimizer.Init(params);

      while (err > tol and iter < 1e5) {
        grad(0) = 2. * (1.5 + params(0) * (params(1) - 1.)) * (params(1) - 1.) +
                  2. * (2.25 + params(0) * (std::pow(params(1), 2) - 1.)) *
                      (std::pow(params(1), 2) - 1.) +
                  2. * (2.625 + params(0) * (std::pow(params(1), 3) - 1.)) *
                      (std::pow(params(1), 3) - 1.) +
                  dist(generator);
        grad(1) = 2. * (1.5 + params(0) * (params(1) - 1.)) * params(0) +
                  2. * (2.25 + params(0) * (std::pow(params(1), 2) - 1.)) * 2. *
                      params(0) * params(0) +
                  2. * (2.625 + params(0) * (std::pow(params(1), 3) - 1.)) *
                      3. * params(0) * std::pow(params(1), 2) +
                  dist(generator);

        optimizer.Update(grad, params);
        err = (params - sol).norm();
        iter += 1;
      }
      REQUIRE(err <= tol);
    }
  }
}

TEST_CASE("optimizers correctly minimize Rosenbrock function", "[optimizer]") {
  auto input_tests = GetOptimizerInputs();
  std::cout << "# Rosenbrock input tests size = " << 7 << std::endl;

  for (std::size_t it = 14; it < 21; it++) {
    std::string name = input_tests[it].dump();

    SECTION("Optimizer test (" + std::to_string(it) + ") on " + name) {
      float err = 1e8;
      float tol = 0.1;
      int iter = 0;

      const double mean = 0.0;
      const double stddev = 1;
      std::default_random_engine generator;
      std::normal_distribution<double> dist(mean, stddev);

      Eigen::VectorXd sol(4);
      sol << 1., 1., 1., 1.;
      Eigen::VectorXd grad(4);
      Eigen::VectorXd params(4);
      params << 4.0, 3.0, 10.0, -5.0;

      netket::Optimizer optimizer(input_tests[it]);
      optimizer.Init(params);

      while (err > tol and iter < 5e5) {
        grad(0) = -400. * params(0) * (params(1) - std::pow(params(0), 2)) -
                  2. * (1 - params(0)) + dist(generator);
        grad(1) = 200. * (params(1) - std::pow(params(0), 2)) -
                  400. * params(1) * (params(2) - std::pow(params(1), 2)) -
                  2. * (1 - params(1)) + dist(generator);
        grad(2) = 200. * (params(2) - std::pow(params(1), 2)) -
                  400. * params(2) * (params(3) - std::pow(params(2), 2)) -
                  2. * (1 - params(2)) + dist(generator);
        grad(3) = 200. * (params(3) - std::pow(params(2), 2)) + dist(generator);

        optimizer.Update(grad, params);
        err = (params - sol).norm();
        iter += 1;
      }
      REQUIRE(err <= tol);
    }
  }
}

TEST_CASE("optimizers correctly minimize Ackley function", "[optimizer]") {
  auto input_tests = GetOptimizerInputs();
  std::cout << "# Ackley input tests size = " << 7 << std::endl;

  for (std::size_t it = 21; it < 28; it++) {
    std::string name = input_tests[it].dump();
    std::string optName = input_tests[it]["Optimizer"]["Name"];
    SECTION("Optimizer test (" + std::to_string(it) + ") on " + name) {
      float err = 1.0e8;
      float tol = 0.1;
      int iter = 0;
      double stddev;

      const double mean = 0.0;
      if ((optName == "RMSProp") || (optName == "AdaDelta")) {
        stddev = 90;
      } else if ((optName == "Sgd") || (optName == "AMSGrad") ||
                 (optName == "Momentum")) {
        stddev = 10;
      } else {
        stddev = 1;
      }
      std::default_random_engine generator;
      std::normal_distribution<double> dist(mean, stddev);

      Eigen::VectorXd sol(2);
      sol << 0, 0;
      Eigen::VectorXd grad(2);
      Eigen::VectorXd params(2);
      params << 2.5, 2.5;

      netket::Optimizer optimizer(input_tests[it]);
      optimizer.Init(params);

      while (err > tol and iter < 5e5) {
        grad(0) =
            2. * std::sqrt(2) * params(0) *
                std::exp(-0.2 * std::sqrt(0.5 * (std::pow(params(0), 2) +
                                                 std::pow(params(1), 2)))) /
                std::sqrt((std::pow(params(0), 2) + std::pow(params(1), 2))) +
            pi * std::sin(2 * pi * params(0)) *
                std::exp(0.5 * (std::cos(2 * pi * params(0)) +
                                std::cos(2 * pi * params(1)))) +
            dist(generator);
        grad(1) =
            2. * std::sqrt(2) * params(1) *
                std::exp(-0.2 * std::sqrt(0.5 * (std::pow(params(0), 2) +
                                                 std::pow(params(1), 2)))) /
                std::sqrt((std::pow(params(0), 2) + std::pow(params(1), 2))) +
            pi * std::sin(2 * pi * params(1)) *
                std::exp(0.5 * (std::cos(2 * pi * params(0)) +
                                std::cos(2 * pi * params(1)))) +
            dist(generator);

        optimizer.Update(grad, params);
        err = (params - sol).norm();
        iter += 1;
      }
      REQUIRE(err <= tol);
    }
  }
}

TEST_CASE("optimizers correctly minimize complex Ackley function",
          "[optimizer]") {
  auto input_tests = GetOptimizerInputs();
  std::cout << "# complex Ackley input tests size = " << 7 << std::endl;

  for (std::size_t it = 28; it < 35; it++) {
    std::string name = input_tests[it].dump();
    std::string optName = input_tests[it]["Optimizer"]["Name"];
    SECTION("Optimizer test (" + std::to_string(it) + ") on " + name) {
      float err = 1.0e8;
      float tol = 0.1;
      int iter = 0;
      double stddev;

      const double mean = 0.0;
      if ((optName == "RMSProp") || (optName == "AdaDelta")) {
        stddev = 90;
      } else if ((optName == "Sgd") || (optName == "AMSGrad") ||
                 (optName == "Momentum")) {
        stddev = 10;
      } else {
        stddev = 1;
      }
      std::default_random_engine generator;
      std::normal_distribution<double> dist(mean, stddev);

      Eigen::VectorXcd sol(2);
      sol << 0.0 + 0.0 * im, 0.0 + 0.0 * im;
      Eigen::VectorXcd grad(2);
      Eigen::VectorXcd params(2);
      params << 0.0 + 2.5 * im, 0.0 + 2.5 * im;

      netket::Optimizer optimizer(input_tests[it]);
      optimizer.Init(params);

      while (err > tol and iter < 5e5) {
        grad(0) =
            0.0 +
            im * (2. * std::sqrt(2) * params(0).imag() *
                      std::exp(
                          -0.2 *
                          std::sqrt(0.5 * (std::pow(params(0).imag(), 2) +
                                           std::pow(params(1).imag(), 2)))) /
                      std::sqrt((std::pow(params(0).imag(), 2) +
                                 std::pow(params(1).imag(), 2))) +
                  pi * std::sin(2 * pi * params(0).imag()) *
                      std::exp(0.5 * (std::cos(2 * pi * params(0).imag()) +
                                      std::cos(2 * pi * params(1).imag()))) +
                  dist(generator));
        grad(1) =
            0.0 +
            im * (2. * std::sqrt(2) * params(1).imag() *
                      std::exp(
                          -0.2 *
                          std::sqrt(0.5 * (std::pow(params(0).imag(), 2) +
                                           std::pow(params(1).imag(), 2)))) /
                      std::sqrt((std::pow(params(0).imag(), 2) +
                                 std::pow(params(1).imag(), 2))) +
                  pi * std::sin(2 * pi * params(1).imag()) *
                      std::exp(0.5 * (std::cos(2 * pi * params(0).imag()) +
                                      std::cos(2 * pi * params(1).imag()))) +
                  dist(generator));

        optimizer.Update(grad, params);
        err = (params - sol).norm();
        iter += 1;
      }
      REQUIRE(err <= tol);
    }
  }
}
