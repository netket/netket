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

#include "catch.hpp"
#include <fstream>
#include <iostream>
#include <limits>

#include "netket.hh"

double loggaussian(double x) { return -(x * x); }

double GaussianWalk(double x, std::mt19937 &gen, double step_size) {
  std::uniform_real_distribution<> dis(0, 1.0);

  double xnew = x + (dis(gen) - 0.5) * step_size;

  if (std::exp(loggaussian(xnew) - loggaussian(x)) > dis(gen)) {
    return xnew;
  } else {
    return x;
  }
}

TEST_CASE("stats miscellanea", "[stats]") {

  std::mt19937 gen(2321);

  netket::Binning<double> binning(16);

  double x = 0;

  std::vector<double> vals;

  int N = 73200;

  for (int i = 0; i < N; i++) {
    x = GaussianWalk(x, gen, 2);
    binning << x;
    vals.push_back(x);
  }

  SECTION("All values are taken into account ") {
    REQUIRE(binning.N() <= N);
    REQUIRE(binning.N() >= int(N / 2));
  }

  SECTION("Averages are between extremes ") {
    double mean = binning.Mean();
    double minval = *std::min_element(vals.begin(), vals.end());
    double maxval = *std::max_element(vals.begin(), vals.end());
    REQUIRE(mean >= minval);
    REQUIRE(mean <= maxval);
  }

  SECTION("Averages are between extremes ") {
    double mean = binning.Mean();
    double minval = *std::min_element(vals.begin(), vals.end());
    double maxval = *std::max_element(vals.begin(), vals.end());
    REQUIRE(mean >= minval);
    REQUIRE(mean <= maxval);
  }

  SECTION("Prevent catastrofic cancellations ") {
    binning.Reset();

    binning << 4;
    binning << 7;
    binning << 13;
    binning << 16;

    double eom1 = binning.ErrorOfMean();

    binning.Reset();

    binning << 1.0e8 + 4;
    binning << 1.0e8 + 7;
    binning << 1.0e8 + 13;
    binning << 1.0e8 + 16;

    double eom2 = binning.ErrorOfMean();

    binning.Reset();

    binning << 1.0e9 + 4;
    binning << 1.0e9 + 7;
    binning << 1.0e9 + 13;
    binning << 1.0e9 + 16;

    double eom3 = binning.ErrorOfMean();

    REQUIRE(Approx(eom2) == eom1);
    REQUIRE(Approx(eom3) == eom1);
  }

  double taucorr1 = binning.TauCorr();

  binning.Reset();
  x = 0;
  for (int i = 0; i < N; i++) {
    x = GaussianWalk(x, gen, 1);
    binning << x;
  }

  double taucorr2 = binning.TauCorr();

  SECTION("Correlation times are consinstent") { REQUIRE(taucorr2 > taucorr1); }
}
