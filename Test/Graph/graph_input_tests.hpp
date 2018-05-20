
#include "Json/json_helper.hpp"
#include <fstream>
#include <string>
#include <vector>

std::vector<json> GetGraphInputs() {

  std::vector<json> input_tests;
  json pars;

  // Hypercube 1d
  pars = {
      {"Graph",
       {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}}};
  input_tests.push_back(pars);

  // Hypercube 2d
  pars = {
      {"Graph",
       {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 2}, {"Pbc", true}}}};
  input_tests.push_back(pars);

  // Hypercube 3d
  pars = {
      {"Graph",
       {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 3}, {"Pbc", true}}}};
  input_tests.push_back(pars);

  // Graph from hilbert space
  pars.clear();
  pars["Hilbert"]["QuantumNumbers"] = {1, -1};
  pars["Hilbert"]["Size"] = 10;

  input_tests.push_back(pars);

  return input_tests;
}
