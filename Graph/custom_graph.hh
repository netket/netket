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

#ifndef NETKET_CUSTOM_GRAPH_HH
#define NETKET_CUSTOM_GRAPH_HH

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <map>
#include <mpi.h>

namespace netket{

/**
    Class for user-defined graphs
    The list of edges and nodes is read from a json input file.
*/
class CustomGraph: public AbstractGraph{

  //adjacency list
  std::vector<std::vector<int>> adjlist_;

  int nsites_;

  int mynode_;

public:

  //Json constructor
  CustomGraph(const json & pars)
  {
    Init(pars);
  }

  void Init(const json & pars){

    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    //Try to construct from explicit graph definition
    if(FieldExists(pars,"Graph")){

      if(FieldExists(pars["Graph"],"AdjacencyList")){
        adjlist_=pars["Graph"]["AdjacencyList"].get<std::vector<std::vector<int>>>();
      }
      if(FieldExists(pars["Graph"],"Edges")){
        std::vector<std::vector<int>> edges=pars["Graph"]["Edges"].get<std::vector<std::vector<int>>>();
        std::cout<<"Here "<<edges.size()<<std::endl;
        AdjacencyListFromEdges(edges);
      }
      if(FieldExists(pars["Graph"],"Size")){
        assert(pars["Graph"]["Size"]>0);
        adjlist_.resize(pars["Graph"]["Size"]);
      }
    }
    else if(FieldExists(pars,"Hilbert")){
      Hilbert hilbert(pars);
      nsites_=hilbert.Size();
      assert(nsites_>0);
      adjlist_.resize(nsites_);
    }
    else {
      if(mynode_==0){
        cerr<<"Graph: one among Size, AdjacencyList, Edges, or Hilbert Space Size must be specified"<<endl;
      }
      std::abort();
    }

    nsites_=adjlist_.size();

    CheckGraph();

    if(mynode_==0){
      std::cout<<"# Graph created "<<std::endl;
      std::cout<<"# Number of nodes = "<<nsites_<<std::endl;
    }
  }


  void AdjacencyListFromEdges(const std::vector<std::vector<int>>& edges){
    nsites_=0;

    for(auto edge : edges){
      if(edge.size()!=2){
        std::cerr<<"# The edge list is invalid"<<std::endl;
        std::abort();
      }
      if(edge[0]<0 || edge[1]<0){
        std::cerr<<"# The edge list is invalid"<<std::endl;
        std::abort();
      }

      nsites_=std::max(std::max(edge[0],edge[1]),nsites_);
    }

    nsites_++;
    adjlist_.resize(nsites_);

    for(auto edge : edges){
      adjlist_[edge[0]].push_back(edge[1]);
      adjlist_[edge[1]].push_back(edge[0]);
    }

  }

  void CheckGraph(){
    for(int i=0;i<nsites_;i++){
      for(auto s : adjlist_[i]){
        //Checking if the referenced nodes are within the expected range
        if(s>=nsites_ || s<0){
          if(mynode_==0){
            std::cerr<<"# The graph is invalid"<<std::endl;
          }
          std::abort();
        }
        //Checking if the adjacency list is symmetric
        //i.e. if site s is declared neihgbor of site i
        //when site i is declared neighbor of site s
        if(std::count(adjlist_[s].begin(),adjlist_[s].end(),i)!=1){
          if(mynode_==0){
            std::cerr<<"# The graph adjacencylist is not symmetric"<<std::endl;
          }
          std::abort();
        }
      }
    }
  }

  //Returns a list of permuted sites equivalent with respect to
  //translation symmetry
  std::vector<std::vector<int>> SymmetryTable()const{

    std::cerr<<"Cannot generate translation symmetries in a custom graph"<<std::endl;
    std::abort();

    std::vector<std::vector<int>> permtable;

    return permtable;
  }

  int Nsites()const{
    return nsites_;
  }


  std::vector<std::vector<int>> AdjacencyList()const{
    return adjlist_;
  }

  bool IsBipartite()const{
    return false;
  }

  //returns the distances of each point from the others
  std::vector<std::vector<int>> Distances()const{
    std::vector<std::vector<int>> distances;

    for(int i=0;i<nsites_;i++){
      distances.push_back(FindDist(adjlist_,i));
    }

    return distances;
  }
};

}
#endif
