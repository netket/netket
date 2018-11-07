
#include <Graph/graph.hpp>
#include <fstream>
#include <string>
#include <vector>
#include "Graph/abstract_graph.hpp"
#include "Utils/memory_utils.hpp"

using Ptype = std::unique_ptr<netket::AbstractGraph>;
std::vector<Ptype> GetGraphs() {
  using namespace netket;
  std::vector<Ptype> graphs;
  {
    netket::Hypercube graph(3, 1, true);
    graphs.push_back(netket::make_unique<Hypercube>(graph));
  }
  {
    netket::Hypercube graph(20, 1, true);
    graphs.push_back(netket::make_unique<Hypercube>(graph));
  }
  {
    netket::Hypercube graph(20, 2, true);
    graphs.push_back(netket::make_unique<Hypercube>(graph));
  }
  {
    netket::Hypercube graph(10, 3, true);
    graphs.push_back(netket::make_unique<Hypercube>(graph));
  }
  {
    netket::CustomGraph graph(14);
    graphs.push_back(netket::make_unique<CustomGraph>(graph));
  }
  {
    netket::CustomGraph graph(0, {{1, 2}, {0, 2}, {0, 1}});
    graphs.push_back(netket::make_unique<CustomGraph>(graph));
  }
  {
    netket::CustomGraph graph(0, {}, {{0, 1}, {0, 2}, {0, 3}, {3, 1}});
    graphs.push_back(netket::make_unique<CustomGraph>(graph));
  }
  // TODO add more CustomGraph constructors
  return graphs;
}
