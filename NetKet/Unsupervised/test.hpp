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

#ifndef NETKET_TEST_HPP_
#define NETKET_TEST_HPP_

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Observable/observable.hpp"
#include "Optimizer/optimizer.hpp"
#include "Sampler/sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include "GroundState/json_output_writer.hpp"
#include "GroundState/matrix_replacement.hpp"

namespace netket {

// Class for unsupervised learning

class Test {
 
  using GsType = std::complex<double>;

  using VectorT =
      Eigen::Matrix<typename Machine<GsType>::StateType, Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename Machine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  std::vector<LocalOperator> operators_;
  Hilbert hilbert_;
  
  Sampler<Machine<GsType>> &sampler_;
  Machine<GsType> &psi_;

  MatrixT Ok_;
  VectorT Okmean_;

  Eigen::MatrixXd vsamp_;
  Eigen::VectorXcd grad_;
  Eigen::VectorXcd rotated_grad_;

  int totalnodes_;
  int mynode_;

  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;

  int npar_;
  
  netket::default_random_engine rgen_;
  
  const std::complex<double> I_;
  Eigen::MatrixXd trainSamples_;
  Eigen::VectorXcd wf_;
  Eigen::MatrixXd basis_states_;
  std::map<std::string,Eigen::MatrixXcd> U_;
  std::vector<std::vector<std::string> > basisSet_;
  std::string basis_;
  
  double KL_;
  double Z_; 
  
  public:
  using MatType = LocalOperator::MatType;

  Test(Sampler<Machine<GsType>> &sampler,const json &pars)
      : hilbert_(pars),
        sampler_(sampler),
        psi_(sampler.Psi()),
        I_(0,1) {
    // DEPRECATED (to remove for v2.0.0)
    if (FieldExists(pars, "Learning")) {
      auto pars1 = pars;
      pars1["Unsupervised"] = pars["Learning"];
      Init(pars1);
    } else {
      Init(pars);
    }
    InitOutput(pars);
  }

  void InitOutput(const json &pars) {
    // DEPRECATED (to remove for v2.0.0)
    auto pars_gs = FieldExists(pars, "Unsupervised") ? pars["Unsupervised"]
                                                    : pars["Learning"];
  }

  void Init(const json &pars) {
    
    
    auto pars_data = pars["Data"];
    auto rotations = pars_data["Rotations"].get<std::vector<MatType>>();
    //auto sites =
    //    pars_data["Sites"].get<std::vector<std::vector<int>>>();
 
    LoadTrainingData();
    LoadWavefunction();
    SetBasisRotations();
    LoadBasisConfiguration();

    basis_states_.resize(1<<psi_.Nvisible(),psi_.Nvisible());
    std::bitset<10> bit;
    for(int i=0;i<1<<psi_.Nvisible();i++){
      bit = i;
      for(int j=0;j<psi_.Nvisible();j++){
        basis_states_(i,j) = 1.0-2.0*bit[psi_.Nvisible()-j-1];
      }
    }
    
    npar_ = psi_.Npar();

    grad_.resize(npar_);
    rotated_grad_.resize(npar_);

    Okmean_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    
    basis_ = FieldVal(pars["Unsupervised"],"Basis","Unsupervised");
    
    nsamples_ = FieldVal(pars["Unsupervised"], "Nsamples", "Unsupervised");

    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void InitSweeps() {
    sampler_.Reset();

    for (int i = 0; i < ninitsamples_; i++) {
      sampler_.Sweep();
    }
  }

  void Sample() {
    sampler_.Reset();

    for (int i = 0; i < ndiscardedsamples_; i++) {
      sampler_.Sweep();
    }

    vsamp_.resize(nsamples_node_, psi_.Nvisible());

    for (int i = 0; i < nsamples_node_; i++) {
      sampler_.Sweep();
      vsamp_.row(i) = sampler_.Visible();
    }
  }

  //Compute the partition function by exact enumeration 
  void ExactPartitionFunction() {
      Z_ = 0.0;
      for(int i=0;i<basis_states_.rows();i++){
          Z_ += std::norm(std::exp(psi_.LogVal(basis_states_.row(i))));
      }
  }

  //Compute KL divergence exactly
  void ExactKL(){
    KL_ = 0.0;
    for(int i=0;i<basis_states_.rows();i++){
      if (std::norm(wf_(i))>0.0){
        KL_ += std::norm(wf_(i))*log(std::norm(wf_(i)));
      }
      KL_ -= std::norm(wf_(i))*log(std::norm(exp(psi_.LogVal(basis_states_.row(i)))));
      KL_ += std::norm(wf_(i))*log(Z_);
    }
  }

  // Test the derivatives of the KL divergence
  void TestDerKL(double eps=0.0000001){
    auto pars = psi_.GetParameters();
    ExactPartitionFunction();
    Eigen::VectorXcd derKL(npar_);
    Eigen::VectorXcd alg_ders;
    Eigen::VectorXcd num_ders_real;
    Eigen::VectorXcd num_ders_imag;
    alg_ders.setZero(npar_);
    num_ders_real.setZero(npar_);
    num_ders_imag.setZero(npar_);
     
    //-- ALGORITHMIC DERIVATIVES --//
    for(int j=0;j<basis_states_.rows();j++){
      alg_ders -= 2.0*std::norm(wf_(j))*psi_.DerLog(basis_states_.row(j)).conjugate();
      alg_ders += 2.0*(std::norm(std::exp(psi_.LogVal(basis_states_.row(j))))/Z_) * psi_.DerLog(basis_states_.row(j)).conjugate();
    }
      
    //-- NUMERICAL DERIVATIVES --//
    for(int p=0;p<npar_;p++){
      pars(p)+=eps;
      psi_.SetParameters(pars);
      double valp=0.0;
      ExactPartitionFunction();
      ExactKL();
      valp = KL_;
      pars(p)-=2.0*eps;
      psi_.SetParameters(pars);
      double valm=0.0;
      ExactPartitionFunction();
      ExactKL();
      valm = KL_;
      pars(p)+=eps;
      num_ders_real(p)=(-valm+valp)/(eps*2.0);

      pars(p)+=I_*eps;
      psi_.SetParameters(pars);
      ExactPartitionFunction();
      ExactKL();
      valp = KL_;
      pars(p)-=I_*2.0*eps;
      psi_.SetParameters(pars);
      ExactPartitionFunction();
      ExactKL();
      valm = KL_;
      pars(p)+=eps;
      num_ders_imag(p)=-(-valm+valp)/(I_*eps*2.0);
      std::cout<<"Numerical Gradient = (";
      std::cout<<num_ders_real(p).real()<<" , "<<num_ders_imag(p).imag()<<")\t-->";
      std::cout<<"(";
      std::cout<<alg_ders(p).real()<<" , "<<alg_ders(p).imag()<<")     ";
      std::cout<<std::endl; 
    }
  }

  void LoadTrainingData(){
    int trainSize = 10000;
    //std::string fileName = "data_tfim10.txt";
    std::string fileName = "qubits_train_samples.txt"; 
    trainSamples_.resize(trainSize,psi_.Nvisible());
    std::ifstream fin_samples(fileName);
    int tmp;
    for (int n=0; n<trainSize; n++) {
      for (int j=0; j<psi_.Nvisible(); j++) {
        fin_samples>> tmp;
        trainSamples_(n,j) = 1.0-2.0*tmp;
      }
    }
  }

  void LoadWavefunction(){
    //std::string fileName = "wf_tfim10.txt";
    std::string fileName = "qubits_psi.txt";
    std::ifstream fin(fileName);
    wf_.resize(1<<psi_.Nvisible());
    double x_in;
    for(int i=0;i<1<<psi_.Nvisible();i++){
      fin >> x_in;
      wf_.real()(i) = x_in;
      fin >> x_in;
      wf_.imag()(i) = x_in;
    }
    //std::cout<<wf_<<std::endl;
  }

  void SetBasisRotations(){
    double oneDivSqrt2 = 1.0/sqrt(2.0);
    
    //X rotation
    U_["X"].setZero(2,2);
    U_["X"].real()<< oneDivSqrt2,oneDivSqrt2,
                    oneDivSqrt2,-oneDivSqrt2;
    //Y rotation
    U_["Y"].resize(2,2);
    U_["Y"].real()<< oneDivSqrt2,0.0,
                    oneDivSqrt2,0.0;
    U_["Y"].imag()<< 0.0,-oneDivSqrt2,
                    0.0,oneDivSqrt2;

  }

  void LoadBasisConfiguration() {
    std::ifstream fin("qubits_bases.txt");
    int num_basis = 5;
    basisSet_.resize(num_basis,std::vector<std::string>(psi_.Nvisible()));
    for (int b=0;b<num_basis;b++){
      for (int j=0;j<psi_.Nvisible(); j++){
        fin >> basisSet_[b][j];
        //std::cout<<basisSet_[b][j]<<" ";
      }
    }
  }

//  void rotateWF(const std::vector<std::string> & basis){
//    int t,counter;
//    std::complex<double> U,Upsi;
//    std::bitset<16> bit;
//    std::bitset<16> st;
//    std::vector<int> basisIndex;
//    Eigen::VectorXd state(psi_.Nvisible());
//    Eigen::VectorXd v(psi_.Nvisible());
//    Eigen::VectorXcd psiR(1<<psi_.Nvisible());
//
//    for(int x=0;x<1<<psi_.Nvisible();x++){
//      U = 1.0;
//      Upsi=0.0;
//      basisIndex.clear();
//      t = 0;
//      st = x;
//      for (int j=0;j<psi_.Nvisible();j++){
//        state(j) = st[psi_.Nvisible()-1-j];
//      }
//      for(int j=0;j<psi_.Nvisible();j++){
//        if (basis[j]!="Z"){
//          t++;
//          basisIndex.push_back(j);
//        }
//      }
//      for(int i=0;i<1<<t;i++){
//        counter  =0;
//        bit = i;
//        v=state;
//        for(int j=0;j<psi_.Nvisible();j++){
//          if (basis[j] != "Z"){
//            v(j) = bit[counter];
//            counter++;
//          }
//        }
//        U=1.0;
//        for(int ii=0;ii<t;ii++){
//          U = U * U_[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(v(basisIndex[ii])));
//        }
//        Upsi += U * std::exp(psi_.LogVal(v));//PSI_.psi(v);
//      }
//      psiR(x) = Upsi;
//    }
//  }
//
//  void rotateWF(const std::vector<std::string> & basis, Eigen::VectorXcd &psiR){
//    int t,counter;
//    std::complex<double> U,Upsi;
//    std::bitset<16> bit;
//    std::bitset<16> st;
//    std::vector<int> basisIndex;
//    Eigen::VectorXd state(psi_.Nvisible());
//    Eigen::VectorXd v(psi_.Nvisible());
//    Eigen::VectorXcd psiR(1<<psi_.Nvisible());
//
//    for(int x=0;x<1<<psi_.Nvisible();x++){
//      U = 1.0;
//      Upsi=0.0;
//      basisIndex.clear();
//      t = 0;
//      st = x;
//      for (int j=0;j<psi_.Nvisible();j++){
//        state(j) = st[psi_.Nvisible()-1-j];
//      }
//      for(int j=0;j<psi_.Nvisible();j++){
//        if (basis[j]!="Z"){
//          t++;
//          basisIndex.push_back(j);
//        }
//      }
//      for(int i=0;i<1<<t;i++){
//        counter  =0;
//        bit = i;
//        v=state;
//        for(int j=0;j<psi_.Nvisible();j++){
//          if (basis[j] != "Z"){
//            v(j) = bit[counter];
//            counter++;
//          }
//        }
//        U=1.0;
//        for(int ii=0;ii<t;ii++){
//          U = U * U_[basis[basisIndex[ii]]](int(state(basisIndex[ii])),int(v(basisIndex[ii])));
//        }
//        Upsi += U * std::exp(psi_.LogVal(v));//PSI_.psi(v);
//      }
//      psiR(x) = Upsi;
//    }
//  }

};

}  // namespace netket

#endif  // NETKET_UNSUPERVISED_HPP
