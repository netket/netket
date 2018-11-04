
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetHilbertInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Spin 1/2
  pars = {{"Graph", {{"Name", "Custom"}, {"Size", 20}}},
          {"Hilbert", {{"Name", "Spin"}, {"S", 0.5}}}};
  input_tests.push_back(pars);

  // Spin 1/2 with total Sz
  pars = {{"Hilbert",
           {{"Name", "Spin"}, {"Size", 20}, {"S", 0.5}, {"TotalSz", 1.0}}}};
  input_tests.push_back(pars);
  //
  // Spin 3
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Size", 25}, {"S", 3}}}};
  input_tests.push_back(pars);

  // Boson
  pars = {{"Hilbert", {{"Name", "Boson"}, {"Size", 21}, {"Nmax", 5}}}};
  input_tests.push_back(pars);

  // Boson
  pars = {{"Graph", {{"Name", "Custom"}, {"Size", 21}}},
          {"Hilbert", {{"Name", "Boson"}, {"Nmax", 5}}}};
  input_tests.push_back(pars);

  // Boson with total number
  pars = {{"Hilbert",
           {{"Name", "Boson"}, {"Size", 21}, {"Nmax", 5}, {"Nbosons", 11}}}};
  input_tests.push_back(pars);
  //
  // Qubit
  pars = {{"Hilbert", {{"Name", "Qubit"}, {"Size", 32}}}};
  input_tests.push_back(pars);

  // Custom Hilbert
  pars = {{"Hilbert", {{"QuantumNumbers", {-1232, 132, 0}}, {"Size", 34}}}};
  input_tests.push_back(pars);

  // // Small hilbert spaces
  // Spin 1/2
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Size", 10}, {"S", 0.5}}}};
  input_tests.push_back(pars);
  //
  // Spin 3
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Size", 4}, {"S", 3}}}};
  input_tests.push_back(pars);

  // Boson
  pars = {{"Hilbert", {{"Name", "Boson"}, {"Size", 5}, {"Nmax", 3}}}};
  input_tests.push_back(pars);

  // Qubit
  pars = {{"Hilbert", {{"Name", "Qubit"}, {"Size", 11}}}};
  input_tests.push_back(pars);

  // Custom Hilbert
  pars = {{"Hilbert", {{"QuantumNumbers", {-1232, 132, 0}}, {"Size", 5}}}};
  input_tests.push_back(pars);

  return input_tests;
}
