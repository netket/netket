
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetOptimizerInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;
  // Check that the optimizers are stepping correctly
  // Minimize Matyas function
  pars = {{"Optimizer", {{"Name", "Sgd"}, {"LearningRate", 0.1}}}};

  input_tests.push_back(pars);

  pars = {
      {"Optimizer", {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 0.1}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"},
            {"Alpha", 0.1},
            {"Beta1", 0.8},
            {"Beta2", 0.95},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AMSGrad"},
            {"LearningRate", 0.1},
            {"Beta1", 0.8},
            {"Beta2", 0.95},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"},
            {"LearningRate", 0.001},
            {"Beta", 0.9},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "Momentum"}, {"LearningRate", 0.1}, {"Beta", 0.95}}}};

  input_tests.push_back(pars);

  // Minimize Beale function
  pars = {{"Optimizer", {{"Name", "Sgd"}, {"LearningRate", 1e-4}}}};

  input_tests.push_back(pars);

  pars = {
      {"Optimizer", {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 1.5}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"},
            {"Alpha", 0.01},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AMSGrad"},
            {"LearningRate", 0.025},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"},
            {"LearningRate", .01},
            {"Beta", 0.9},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "Momentum"}, {"LearningRate", 4e-5}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);

  // Minimize Rosenbrock function
  pars = {{"Optimizer", {{"Name", "Sgd"}, {"LearningRate", 4e-5}}}};

  input_tests.push_back(pars);

  pars = {
      {"Optimizer", {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 24}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"},
            {"Alpha", 0.01},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AMSGrad"},
            {"LearningRate", 0.5},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"},
            {"LearningRate", .01},
            {"Beta", 0.9},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "Momentum"}, {"LearningRate", 2.5e-4}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);

  // Minimize Ackley function
  pars = {{"Optimizer", {{"Name", "Sgd"}, {"LearningRate", .005}}}};

  input_tests.push_back(pars);

  pars = {
      {"Optimizer", {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-5}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 2}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"},
            {"Alpha", 1},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AMSGrad"},
            {"LearningRate", 0.1},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"},
            {"LearningRate", .01},
            {"Beta", 0.9},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "Momentum"}, {"LearningRate", .005}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);

  // Minimize complex function
  pars = {{"Optimizer", {{"Name", "Sgd"}, {"LearningRate", .005}}}};

  input_tests.push_back(pars);

  pars = {
      {"Optimizer", {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-5}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 2}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"},
            {"Alpha", 1},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AMSGrad"},
            {"LearningRate", 0.1},
            {"Beta1", 0.9},
            {"Beta2", 0.999},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"},
            {"LearningRate", .01},
            {"Beta", 0.9},
            {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "Momentum"}, {"LearningRate", .005}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);

  return input_tests;
}
