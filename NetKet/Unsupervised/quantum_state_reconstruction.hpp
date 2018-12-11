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

#ifndef NETKET_QUANTUMSTATERECONSTRUCTION_HPP_
#define NETKET_QUANTUMSTATERECONSTRUCTION_HPP_

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "Machine/machine.hpp"
#include "Operator/abstract_operator.hpp"
#include "Optimizer/optimizer.hpp"
#include "Output/json_output_writer.hpp"
#include "Sampler/abstract_sampler.hpp"
#include "Stats/stats.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/random_utils.hpp"
#include <bitset>
namespace netket {

// Class for unsupervised learningclass Test {
 
class QuantumStateReconstruction {

  using GsType = std::complex<double>;
  using VectorT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename AbstractMachine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  AbstractSampler<AbstractMachine<GsType>> &sampler_;
  AbstractMachine<GsType> &psi_;
  const AbstractHilbert &hilbert_;
  AbstractOptimizer &opt_;

  std::vector<LocalOperator> rotations_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  MatrixT Ok_;
  VectorT Okmean_;
  
  Eigen::MatrixXd vsamp_;
  Eigen::VectorXcd grad_;
  Eigen::VectorXcd rotated_grad_;
  Eigen::VectorXcd gradprev_;
  Eigen::MatrixXcd gradfull_;

  double sr_diag_shift_;
  bool sr_rescale_shift_;
  bool use_iterative_;

  // This optional will contain a value iff the MPI rank is 0.
  nonstd::optional<JsonOutputWriter> output_;

  int totalnodes_;
  int mynode_;

  bool dosr_;
  bool use_cholesky_;
  int batchsize_; 
  int batchsize_node_;
  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;
  int niter_opt_;

  int npar_;
  
  netket::default_random_engine rgen_;
  
  std::vector<Eigen::VectorXd> trainingSamples_;
  std::vector<int> trainingBases_;
  Eigen::VectorXcd wf_;
  //std::vector<Eigen::VectorXcd> rotated_wf_;
  Eigen::MatrixXd basis_states_;
 
  double overlap_;
  //double KL_;
  double logZ_; 
  //double NLL_;
  //const std::complex<double> I_;

  public:
  using MatType = LocalOperator::MatType;

  QuantumStateReconstruction( AbstractSampler<AbstractMachine<GsType>> &sampler,
                              AbstractOptimizer &opt,
                              int batchsize,
                              int nsamples,
                              int niter_opt,
                              std::vector<MatType> jrot,
                              std::vector<std::vector<int> > sites,
                              std::vector<Eigen::VectorXd> trainingSamples,
                              std::vector<int> trainingBases,
                              int ndiscardedsamples=-1,
                              int discarded_samples_on_init=0,
                              std::string method = "Gd",
                              double diagshift = 0.01,
                              bool rescale_shift = false,
                              bool use_iterative = false,
                              bool use_cholesky = true
                              )
      : sampler_(sampler),
        psi_(sampler_.GetMachine()),
        hilbert_(psi_.GetHilbert()),
        opt_(opt),
        trainingSamples_(trainingSamples),
        trainingBases_(trainingBases){
        //  ,I_(0,1){
   
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());        
          
    grad_.resize(npar_);
    rotated_grad_.resize(npar_);

    Okmean_.resize(npar_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    
    batchsize_ = batchsize;
    batchsize_node_ = int(std::ceil(double(batchsize_) / double(totalnodes_)));
 
    nsamples_ = nsamples;
    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));
   
    ninitsamples_ = discarded_samples_on_init;

    if (ndiscardedsamples == -1) {
      ndiscardedsamples_ = 0.1 * nsamples_node_;
    } else {
      ndiscardedsamples_ = ndiscardedsamples;
    }

    niter_opt_ = niter_opt;

    if (method == "Gd") {
      dosr_ = false;
    } else {
      setSrParameters(diagshift, rescale_shift, use_iterative, use_cholesky);
    }

    if (dosr_) {
      InfoMessage() << "Using the Stochastic reconfiguration method"
                    << std::endl;

      if (use_iterative_) {
        InfoMessage() << "With iterative solver" << std::endl;
      } else {
        if (use_cholesky_) {
          InfoMessage() << "Using Cholesky decomposition" << std::endl;
        }
      }
    } else {
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    }
    
    //TODO Change this hack 
    for(std::size_t i=0; i<jrot.size(); i++){
      if (sites[i].size() == 0){
        LocalOperator::SiteType vec(1,0);
        LocalOperator::MatType id(2);
        id[0].resize(2);
        id[1].resize(2);
        id[0][1] = 0.0;
        id[1][0] = 0.0;
        id[0][0] = 1.0;
        id[1][1] = 1.0;
        rotations_.push_back(LocalOperator(hilbert_,id,vec));
      }
      else{
        rotations_.push_back(LocalOperator(hilbert_, jrot[i], sites[i]));
      }
    }

    InfoMessage() << "Quantum state reconstruction running on " << totalnodes_
                  << " processes" << std::endl;
 
    basis_states_.resize(1<<psi_.Nvisible(),psi_.Nvisible());
    std::bitset<10> bit;
    for(int i=0;i<1<<psi_.Nvisible();i++){
      bit = i;
      for(int j=0;j<psi_.Nvisible();j++){
        //basis_states_(i,j) = 1.0-2.0*bit[psi_.Nvisible()-j-1];
        basis_states_(i,j) = bit[psi_.Nvisible()-j-1];
      }
    }
    
    LoadWavefunction();
    
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

  void Gradient(std::vector<Eigen::VectorXd> &batchSamples,std::vector<int> &batchBases) {
    gradfull_.resize(batchsize_node_+nsamples_node_,psi_.Npar());
    Eigen::VectorXcd der(psi_.Npar());
    Eigen::VectorXcd grad_tmp(psi_.Npar());
    
    // Positive phase driven by data
    const int ndata = batchsize_node_;
    Ok_.resize(ndata, psi_.Npar());
    for (int i = 0; i < ndata; i++) {
      RotateGradient(batchBases[i],batchSamples[i],der);
      Ok_.row(i) = der.conjugate();
      gradfull_.row(i) = -2.0*der.conjugate();
    }
    grad_ = -2.0*(Ok_.colwise().mean());

    // Negative phase driven by the machine
    Sample();

    const int nsamp = vsamp_.rows();
    Ok_.resize(nsamp, psi_.Npar());

    for (int i = 0; i < nsamp; i++) {
      Ok_.row(i) = psi_.DerLog(vsamp_.row(i)).conjugate();
      gradfull_.row(ndata+i) = 2.0*psi_.DerLog(vsamp_.row(i)).conjugate();
    }
    grad_ += 2.0*(Ok_.colwise().mean());
    
    // Summing the gradient over the nodes
    SumOnNodes(gradfull_); 
    gradfull_ /= double(totalnodes_);
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  void Run(){
    std::vector<Eigen::VectorXd> batchSamples;
    std::vector<int> batchBases;
    opt_.Reset();

    InitSweeps();
    std::uniform_int_distribution<int> distribution(0,trainingSamples_.size()-1);

    for (int i = 0; i < niter_opt_; i++) {
      int index;
      batchSamples.resize(batchsize_node_);
      batchBases.resize(batchsize_node_);

      for(int k=0;k<batchsize_node_;k++){
        index = distribution(rgen_);
        batchSamples[k] = trainingSamples_[index];
        batchBases[k] = trainingBases_[index];
      }

      Gradient(batchSamples,batchBases);
      UpdateParameters();
      Scan(i);
    }
  }

  void UpdateParameters() {
    auto pars = psi_.GetParameters();
    Ok_.resize(batchsize_node_+nsamples_node_,psi_.Npar());
    Ok_ = gradfull_;
    if (dosr_) {
      const int nsamp = vsamp_.rows();

      //Eigen::VectorXcd b = Ok_.adjoint() * elocs_;
      //SumOnNodes(b);
      //b /= double(nsamp * totalnodes_);
      Eigen::VectorXcd b = grad_; 
      
      if (!use_iterative_) {
        // Explicit construction of the S matrix
        Eigen::MatrixXcd S = Ok_.adjoint() * Ok_;
        //Eigen::MatrixXcd S = gradfull_.adjoint() * gradfull_;
        SumOnNodes(S);
        S /= double(nsamp * totalnodes_);
        //S /= double(gradfull_.rows());
        // Adding diagonal shift
        S += Eigen::MatrixXd::Identity(pars.size(), pars.size()) *
             sr_diag_shift_;

        Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> qr(S.rows(), S.cols());
        qr.setThreshold(1.0e-6);
        qr.compute(S);
        const Eigen::VectorXcd deltaP = qr.solve(b);
        // Eigen::VectorXcd deltaP=S.jacobiSvd(ComputeThinU |
        // ComputeThinV).solve(b);

        assert(deltaP.size() == grad_.size());
        grad_ = deltaP;

        if (sr_rescale_shift_) {
          std::complex<double> nor = (deltaP.dot(S * deltaP));
          grad_ /= std::sqrt(nor.real());
        }

      } else {
        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper,
                                 Eigen::IdentityPreconditioner>
            it_solver;
        // Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>
        // it_solver;
        it_solver.setTolerance(1.0e-3);
        MatrixReplacement S;
        S.attachMatrix(Ok_);
        //S.attachMatrix(gradfull_);
        S.setShift(sr_diag_shift_);
        S.setScale(1. / double(nsamp));
        //S.setScale(1. / double(gradfull_.rows()));
        //S.setScale(1. / double(OK_.rows()));
        it_solver.compute(S);
        auto deltaP = it_solver.solve(b);

        grad_ = deltaP;
        if (sr_rescale_shift_) {
          auto nor = deltaP.dot(S * deltaP);
          grad_ /= std::sqrt(nor.real());
        }

        // if(mynode_==0){
        //   std::cerr<<it_solver.iterations()<<"
        //   "<<it_solver.error()<<std::endl;
        // }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    opt_.Update(grad_, pars);

    SendToAll(pars);

    psi_.SetParameters(pars);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //Compute different estimators for the training performance
  void Scan(int i){//,Eigen::MatrixXd &nll_test,std::ofstream &obs_out){
    if (mynode_==0) {
      ExactPartitionFunction();
      Overlap();
      PrintStats(i);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Compute the overlap with the target wavefunction
  void Overlap(){
      std::complex<double> tmp;
      for(int i=0;i<basis_states_.rows();i++){
          tmp += std::conj(wf_(i))*std::exp(psi_.LogVal(basis_states_.row(i))-0.5*logZ_);//))/std::sqrt(Z_);
      }
      overlap_ = std::sqrt(std::norm(tmp));
  }
  //Print observer
  void PrintStats(int i){
    InfoMessage() << "Epoch: " << i << " \t";     
    InfoMessage() << "Overlap = " << std::setprecision(10) << overlap_<< "\t";//<< Fcheck_;
    InfoMessage() << std::endl;
  } 

  void setSrParameters(double diagshift = 0.01, bool rescale_shift = false,
                       bool use_iterative = false, bool use_cholesky = true) {
    sr_diag_shift_ = diagshift;
    sr_rescale_shift_ = rescale_shift;
    use_iterative_ = use_iterative;
    dosr_ = true;
    use_cholesky_ = use_cholesky;
  }

  //Compute the partition function by exact enumeration 
  void ExactPartitionFunction() {
      logZ_=0.0;  
      double tmp = 0.0;
      double psi_max = -1.0;
      for(int i=0;i<basis_states_.rows();i++){
        if (psi_.LogVal(basis_states_.row(i)).real()>psi_max){
          psi_max = psi_.LogVal(basis_states_.row(i)).real();
        }
      }
      for(int i=0;i<basis_states_.rows();i++){
        tmp += std::norm(std::exp(psi_.LogVal(basis_states_.row(i))-psi_max));
      }
      logZ_ = std::log(tmp) + 2*psi_max;
  }

  void RotateGradient(int b_index,const Eigen::VectorXd & state,Eigen::VectorXcd &rotated_gradient) {

    std::complex<double> U,den;
    Eigen::VectorXcd num;
    Eigen::VectorXd v(1<<psi_.Nvisible());
    rotations_[b_index].FindConn(state,mel_,connectors_,newconfs_);
    assert(connectors_.size() == mel_.size());
    
    const std::size_t nconn = connectors_.size();
    
    auto logvaldiffs = (psi_.LogValDiff(state, connectors_, newconfs_));  
    den = 0.0;
    num.setZero(psi_.Npar());
    for(std::size_t k=0;k<nconn;k++){
      v = state;
      for (std::size_t j=0; j<connectors_[k].size();j++){
        v(connectors_[k][j]) = newconfs_[k][j];
      }
      num += mel_[k]*std::exp(logvaldiffs(k)) * psi_.DerLog(v);
      den += mel_[k]*std::exp(logvaldiffs(k));
    }
    rotated_gradient = (num/den);
  }

  void LoadWavefunction(){
    std::string fileName = "ising1d_psi.txt";
    std::ifstream fin(fileName);
    wf_.resize(1<<psi_.Nvisible());
    double x_in;
    for(int i=0;i<1<<psi_.Nvisible();i++){
      fin >> x_in;
      wf_.real()(i) = x_in;
      fin >> x_in;
      wf_.imag()(i) = x_in;
    }
  }









  double NLL(std::vector<Eigen::VectorXd> &data_samples,std::vector<int> &data_bases){
    double NLL = 0.0;
    std::complex<double> rotated_psi;
    for(std::size_t i=0; i<data_samples.size(); i++){
      RotateWavefunction(data_bases[i],data_samples[i],rotated_psi);
      NLL -= std::log(std::norm(rotated_psi));
    }
    NLL /= float(data_samples.size()); 
    return NLL;
  }

  void RotateWavefunction(int b_index,Eigen::VectorXd &state,std::complex<double> &psiR) {

    std::complex<double> U,tmp,logpsiR;
    logpsiR = 0.0;

    rotations_[b_index].FindConn(state,mel_,connectors_,newconfs_);
    assert(connectors_.size() == mel_.size());
    
    const std::size_t nconn = connectors_.size();
    
    auto logvaldiffs = (psi_.LogValDiff(state, connectors_, newconfs_));  
    tmp = 0.0;

    for(std::size_t k=0;k<nconn;k++){
      tmp += mel_[k]*std::exp(logvaldiffs(k));
    }
    logpsiR = std::log(tmp) + psi_.LogVal(state);
    psiR = std::exp(logpsiR-0.5*logZ_);
  }


  // Test the derivatives of the KL divergence
  void TestDerNLLsampling(double eps=0.0000001){
    
    std::cout<<"-- Testing derivates of Negative Log-Likelihood with sampling--"<<std::endl;
    const std::complex<double> I_(0.0,1.0);
    double nll;
    auto pars = psi_.GetParameters();
    ExactPartitionFunction();
    Eigen::VectorXcd derNLL(npar_);
    Eigen::VectorXcd alg_ders;
    Eigen::VectorXcd num_ders_real;
    Eigen::VectorXcd num_ders_imag;
    alg_ders.setZero(npar_);
    num_ders_real.setZero(npar_);
    num_ders_imag.setZero(npar_);
     
    //-- ALGORITHMIC DERIVATIVES --//
    for(std::size_t i=0; i<trainingSamples_.size(); i++){
      RotateGradient(trainingBases_[i],trainingSamples_[i],derNLL);
      alg_ders -= 2.0*derNLL.conjugate()/double(trainingSamples_.size()); 
    }
    
    nsamples_node_ = 10000;
    Sample();
    int nsamp = vsamp_.rows();
    for(int i=0; i<nsamp; i++){
      alg_ders += 2.0 * psi_.DerLog(vsamp_.row(i)).conjugate() / double(nsamp);
    }

    //-- NUMERICAL DERIVATIVES --//
    for(int p=0;p<npar_;p++){
      pars(p)+=eps;
      psi_.SetParameters(pars);
      double valp=0.0;
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valp = nll;
      pars(p)-=2.0*eps;
      psi_.SetParameters(pars);
      double valm=0.0;
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valm = nll;
      pars(p)+=eps;
      num_ders_real(p)=(-valm+valp)/(eps*2.0);

      pars(p)+=I_*eps;
      psi_.SetParameters(pars);
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valp = nll;
      pars(p)-=I_*2.0*eps;
      psi_.SetParameters(pars);
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valm = nll;
      pars(p)+=eps;
      num_ders_imag(p)=-(-valm+valp)/(I_*eps*2.0);
      std::cout<<"Numerical Gradient = (";
      std::cout<<num_ders_real(p).real()<<" , "<<num_ders_imag(p).imag()<<")\t-->";
      std::cout<<"(";
      std::cout<<alg_ders(p).real()<<" , "<<alg_ders(p).imag()<<")     ";
      std::cout<<std::endl; 
    }
  }
  // Test the derivatives of the KL divergence
  void TestDerNLL(double eps=0.0000001){
    
    std::cout<<"-- Testing derivates of Negative Log-Likelihood --"<<std::endl;
    const std::complex<double> I_(0.0,1.0);
    double nll;
    auto pars = psi_.GetParameters();
    ExactPartitionFunction();
    Eigen::VectorXcd derNLL(npar_);
    Eigen::VectorXcd alg_ders;
    Eigen::VectorXcd num_ders_real;
    Eigen::VectorXcd num_ders_imag;
    alg_ders.setZero(npar_);
    num_ders_real.setZero(npar_);
    num_ders_imag.setZero(npar_);
     
    //-- ALGORITHMIC DERIVATIVES --//
    for(std::size_t i=0; i<trainingSamples_.size(); i++){
      RotateGradient(trainingBases_[i],trainingSamples_[i],derNLL);
      alg_ders -= 2.0*derNLL.conjugate()/float(trainingSamples_.size()); 
    }
    for(int j=0;j<basis_states_.rows();j++){
      alg_ders += 2.0*(std::norm(std::exp(psi_.LogVal(basis_states_.row(j))-0.5*logZ_))) * psi_.DerLog(basis_states_.row(j)).conjugate();
    }
    //-- NUMERICAL DERIVATIVES --//
    for(int p=0;p<npar_;p++){
      pars(p)+=eps;
      psi_.SetParameters(pars);
      double valp=0.0;
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valp = nll;
      pars(p)-=2.0*eps;
      psi_.SetParameters(pars);
      double valm=0.0;
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valm = nll;
      pars(p)+=eps;
      num_ders_real(p)=(-valm+valp)/(eps*2.0);

      pars(p)+=I_*eps;
      psi_.SetParameters(pars);
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valp = nll;
      pars(p)-=I_*2.0*eps;
      psi_.SetParameters(pars);
      ExactPartitionFunction();
      nll = NLL(trainingSamples_,trainingBases_);
      valm = nll;
      pars(p)+=eps;
      num_ders_imag(p)=-(-valm+valp)/(I_*eps*2.0);
      std::cout<<"Numerical Gradient = (";
      std::cout<<num_ders_real(p).real()<<" , "<<num_ders_imag(p).imag()<<")\t-->";
      std::cout<<"(";
      std::cout<<alg_ders(p).real()<<" , "<<alg_ders(p).imag()<<")     ";
      std::cout<<std::endl; 
    }
  }



};

}  // namespace netket

#endif  // NETKET_UNSUPERVISED_HPP
