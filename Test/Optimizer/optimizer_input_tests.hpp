
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetOptimizerInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;
  // Check that the optimizers are stepping correctly
  // Minimize Matyas function
  pars = {{"Optimizer",
          {{"Name", "Sgd"}, {"LearningRate", 0.1}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 0.1}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"}, {"Alpha", 0.1}, {"Beta1", 0.8}, {"Beta2",0.95}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AMSGrad"}, {"LearningRate", 0.1}, {"Beta1", 0.8}, {"Beta2",0.95}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"}, {"LearningRate", 0.001},  {"Beta", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "Momentum"}, {"LearningRate", 0.1}, {"Beta", 0.95}}}};

  input_tests.push_back(pars);


  // Minimize Beale function
  pars = {{"Optimizer", //Edit increased for Beale
          {{"Name", "Sgd"}, {"LearningRate", 1e-4}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 1.5}, {"Epscut",1.0e-8}}}}; //EDIT increased for Beale

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"}, {"Alpha", 0.01}, {"Beta1", 0.9}, {"Beta2",0.999}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",//Edit decreased for Beale
           {{"Name", "AMSGrad"}, {"LearningRate", 0.025}, {"Beta1", 0.9}, {"Beta2",0.999}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"}, {"LearningRate", .01},  {"Beta", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer", //Increased from 2 to 3 to 4 for Beale
           {{"Name", "Momentum"}, {"LearningRate", 4e-5}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);

  // Minimize Rosenbrock function
  pars = {{"Optimizer",//3.5 to 4 to 5 to 4 Rosenbrock
          {{"Name", "Sgd"}, {"LearningRate", 4e-5}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaDelta"}, {"Rho", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",//22 to 25 to 22 to 23 to 24 Rosenbrock
           {{"Name", "AdaGrad"}, {"LearningRate", 24}, {"Epscut",1.0e-8}}}}; //EDIT increased for Beale

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaMax"}, {"Alpha", 0.01}, {"Beta1", 0.9}, {"Beta2",0.999}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",//Increased Rosenbrock
           {{"Name", "AMSGrad"}, {"LearningRate", 0.5}, {"Beta1", 0.9}, {"Beta2",0.999}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"}, {"LearningRate", .01},  {"Beta", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer", // 5e-4 to 2.5e-4 Rosenbrock
           {{"Name", "Momentum"}, {"LearningRate", 2.5e-4}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);

  // Minimize Ackley function
  pars = {{"Optimizer",
          {{"Name", "Sgd"}, {"LearningRate", .1}}}};

  input_tests.push_back(pars);

//  pars = {{"Optimizer", //0.9 to 0.8 to 0.5 to .25 Ackley
//           {{"Name", "AdaDelta"}, {"Rho", 0.99}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "AdaGrad"}, {"LearningRate", 2}, {"Epscut",1.0e-8}}}}; //EDIT increased for Beale

  input_tests.push_back(pars);

  pars = {{"Optimizer", // 1 to 2 Ackley
           {{"Name", "AdaMax"}, {"Alpha", 1}, {"Beta1", 0.9}, {"Beta2",0.999}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",//.01 to .1 to 0.5 to .25 to 0.35 to .45 to .4 Ackley
           {{"Name", "AMSGrad"}, {"LearningRate", 0.4}, {"Beta1", 0.9}, {"Beta2",0.999}, {"Epscut",1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer",
           {{"Name", "RMSProp"}, {"LearningRate", 1},  {"Beta", 0.9}, {"Epscut", 1.0e-8}}}};

  input_tests.push_back(pars);

  pars = {{"Optimizer", //.1 to .5 to 1 to .75 Ackley
           {{"Name", "Momentum"}, {"LearningRate", .75}, {"Beta", 0.9}}}};

  input_tests.push_back(pars);
  return input_tests;

}
