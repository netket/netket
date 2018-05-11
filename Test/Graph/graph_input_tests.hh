
#include "Json/json.hh"
#include <vector>
#include <string>
#include <fstream>

std::vector<json> GetGraphInputs(){

  std::vector<json> input_tests;
  json pars;

  //Hypercube 1d
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 1},
      {"Pbc", true}
    }}
  };
  input_tests.push_back(pars);

  //Hypercube 2d
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 2},
      {"Pbc", true}
    }}
  };
  input_tests.push_back(pars);

  //Hypercube 3d
  pars={
    {"Graph",{
      {"Name", "Hypercube"},
      {"L", 20},
      {"Dimension", 3},
      {"Pbc", true}
    }}
  };
  input_tests.push_back(pars);

  return input_tests;
}
