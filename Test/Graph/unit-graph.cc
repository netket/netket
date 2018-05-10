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

#include "netket.hh"
#include "catch.hpp"
#include <iostream>
#include <fstream>
#include <vector>

TEST_CASE( "graphs have consistent number of sites", "[graph]" ) {

  int ntests=3;

  std::vector<std::string> inputjson(ntests);

  inputjson[0]="{\"Graph\": {\"Name\": \"Hypercube\", \"L\": 5, \"Dimension\": 1, \"Pbc\": true}}";
  inputjson[1]="{\"Graph\": {\"Name\": \"Hypercube\", \"L\": 5, \"Dimension\": 2, \"Pbc\": true}}";
  inputjson[2]="{\"Graph\": {\"Name\": \"Hypercube\", \"L\": 5, \"Dimension\": 3, \"Pbc\": true}}";

  for(int i=0;i<ntests;i++){
    std::string filename="Graph/test"+std::to_string(i+1)+".json";
    SECTION( "Graph test on "+filename ) {

      auto pars=json::parse(inputjson[i]);

      netket::Graph graph(pars);

      REQUIRE( graph.Nsites() > 0 );
    }
  }

}
