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

#ifndef NETKET_UNSUPERVISED_HPP
#define NETKET_UNSUPERVISED_HPP

//#include <memory>
//
//#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"
//#include "Observable/observable.hpp"
//#include "Optimizer/optimizer.hpp"

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

class ContrastiveDivergence {
 
  using GsType = std::complex<double>;

  using VectorT =
      Eigen::Matrix<typename Machine<GsType>::StateType, Eigen::Dynamic, 1>;
  using MatrixT = Eigen::Matrix<typename Machine<GsType>::StateType,
                                Eigen::Dynamic, Eigen::Dynamic>;

  Hamiltonian &ham_;
  Sampler<Machine<GsType>> &sampler_;
  Machine<GsType> &psi_;

  std::vector<std::vector<int>> connectors_;
  std::vector<std::vector<double>> newconfs_;
  std::vector<std::complex<double>> mel_;

  Eigen::VectorXcd elocs_;
  MatrixT Ok_;
  VectorT Okmean_;

  Eigen::MatrixXd vsamp_;

  Eigen::VectorXcd grad_;
  Eigen::VectorXcd gradprev_;

  double sr_diag_shift_;
  bool sr_rescale_shift_;
  bool use_iterative_;

  int totalnodes_;
  int mynode_;

  // This optional will contain a value iff the MPI rank is 0.
  nonstd::optional<JsonOutputWriter> output_;

  Optimizer &opt_;

  std::vector<Observable> obs_;
  ObsManager obsmanager_;

  bool dosr_;


  int batchsize_; //TODO
  int batchsize_node_;
  int cd_; //TODO
  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;
  int niter_opt_;

  std::complex<double> elocmean_;
  double elocvar_;
  int npar_;
  
  netket::default_random_engine rgen_;
  
  //TODO TEMPORARY STUFF
  const std::complex<double> I_;
  Eigen::MatrixXd trainSamples_;
  Eigen::VectorXd wf_;
  Eigen::MatrixXd basis_states_;
  double fidelity_;
  double KL_;
  double Z_; 
  
  public:

  ContrastiveDivergence(Hamiltonian &ham, Sampler<Machine<GsType>> &sampler,
                        Optimizer &opt, const json &pars)
      : ham_(ham),
        sampler_(sampler),
        psi_(sampler.Psi()),
        opt_(opt),
        obs_(Observable::FromJson(ham.GetHilbert(), pars)),
        elocvar_(0.),
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
    if (mynode_ == 0) {
      output_ = JsonOutputWriter::FromJson(pars_gs);
    }
  }

  void Init(const json &pars) {
    
    //TODO TEMPOERARY STUFF 
    LoadTrainingData();
    LoadWavefunction();
    basis_states_.resize(1<<psi_.Nvisible(),psi_.Nvisible());
    std::bitset<10> bit;
    for(int i=0;i<1<<psi_.Nvisible();i++){
      bit = i;
      for(int j=0;j<psi_.Nvisible();j++){
        basis_states_(i,j) = 1.0-2.0*bit[psi_.Nvisible()-j-1];
      }
    }
    //TODO 
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);
    Okmean_.resize(npar_);

    setSrParameters();

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    cd_ = 10;//FieldVal(pars["Unsupervised"], "CDsteps", "Unsupervised");

    batchsize_ = 100;//FieldVal(pars["Unsupervised"], "Batchsize", "Unsupervised");
    batchsize_node_ = int(std::ceil(double(batchsize_) / double(totalnodes_)));

    nsamples_ = FieldVal(pars["Unsupervised"], "Nsamples", "Unsupervised");

    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    ninitsamples_ =
        FieldOrDefaultVal(pars["Unsupervised"], "DiscardedSamplesOnInit", 0.);

    ndiscardedsamples_ = FieldOrDefaultVal(
        pars["Unsupervised"], "DiscardedSamples", 0.1 * nsamples_node_);

    niter_opt_ = FieldVal(pars["Unsupervised"], "NiterOpt", "Unsupervised");

    if (pars["Unsupervised"]["Method"] == "Gd") {
      dosr_ = false;
    } else {
      double diagshift =
          FieldOrDefaultVal(pars["Unsupervised"], "DiagShift", 0.01);
      bool rescale_shift =
          FieldOrDefaultVal(pars["Unsupervised"], "RescaleShift", false);
      bool use_iterative =
          FieldOrDefaultVal(pars["Unsupervised"], "UseIterative", false);

      setSrParameters(diagshift, rescale_shift, use_iterative);
    }

    if (dosr_) {
      InfoMessage() << "Using the Stochastic reconfiguration method"
                    << std::endl;
      if (use_iterative_) {
        InfoMessage() << "With iterative solver" << std::endl;
      }
    } else {
      InfoMessage() << "Using a gradient-descent based method" << std::endl;
    }

    InfoMessage() << "Unsupervised learning running on " << totalnodes_
                  << " processes" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void InitSweeps() {
    sampler_.Reset();

    for (int i = 0; i < ninitsamples_; i++) {
      sampler_.Sweep();
    }
  }

  void Sample_VMC() {
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

  void Sample() {
    sampler_.Reset();
    vsamp_.resize(nsamples_node_, psi_.Nvisible());
    int index;
    
    //TODO Check the initialization of the distribution
    std::uniform_int_distribution<int> distribution(0,trainSamples_.rows()-1);
    
    for (int i = 0; i < nsamples_node_; i++) {
      //TODO Check this loop
      index = distribution(rgen_);
      sampler_.SetVisible(trainSamples_.row(index));
      for (int j=0; j<cd_; j++){
        sampler_.Sweep();
      }
      vsamp_.row(i) = sampler_.Visible();
    }
  }

  void Gradient(const Eigen::MatrixXd & batchSamples) {
    obsmanager_.Reset("Energy");
    obsmanager_.Reset("EnergyVariance");

    for (const auto &ob : obs_) {
      obsmanager_.Reset(ob.Name());
    }

    // Positive phase driven by data
    const int ndata = batchsize_node_;
    Ok_.resize(ndata, psi_.Npar());
    for (int i = 0; i < ndata; i++) {
      Ok_.row(i) = psi_.DerLog(batchSamples.row(i));
    }
    grad_ = -2.0*(Ok_.colwise().mean()).conjugate();
    
    //TODO Negative phase exact
    //ExactPartitionFunction();
    //for(int j=0;j<basis_states_.rows();j++){
    //  grad_ += 2.0*(std::norm(std::exp(psi_.LogVal(basis_states_.row(j))))/Z_) * (psi_.DerLog(basis_states_.row(j))).conjugate();
    //}
    
    // Negative phase driven by the machine
    Sample_VMC();

    const int nsamp = vsamp_.rows();
    elocs_.resize(nsamp);
    Ok_.resize(nsamp, psi_.Npar());

    for (int i = 0; i < nsamp; i++) {
      elocs_(i) = Eloc(vsamp_.row(i));
      Ok_.row(i) = psi_.DerLog(vsamp_.row(i));
      obsmanager_.Push("Energy", elocs_(i).real());

      for (const auto &ob : obs_) {
        obsmanager_.Push(ob.Name(), ObSamp(ob, vsamp_.row(i)));
      }
    }
    elocmean_ = elocs_.mean();
    SumOnNodes(elocmean_);
    elocmean_ /= double(totalnodes_);

    elocs_ -= elocmean_ * Eigen::VectorXd::Ones(nsamp);
    for (int i = 0; i < nsamp; i++) {
      obsmanager_.Push("EnergyVariance", std::norm(elocs_(i)));
    }

    grad_ += 2.0*(Ok_.colwise().mean()).conjugate();
    
    // Summing the gradient over the nodes
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_);
  }

  std::complex<double> Eloc(const Eigen::VectorXd &v) {
    ham_.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    std::complex<double> eloc = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      eloc += mel_[i] * std::abs(std::exp(logvaldiffs(i)));
    }

    return eloc;
  }

  double ObSamp(const Observable &ob, const Eigen::VectorXd &v) {
    ob.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    std::complex<double> obval = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      obval += mel_[i] * std::abs(std::exp(logvaldiffs(i)));
    }

    return obval.real();
  }

  double ElocMean() { return elocmean_.real(); }

  double Elocvar() { return elocvar_; }

//  void Run(const Eigen::MatrixXd & trainSamples) {
  void Run(){
    Eigen::MatrixXd batch_samples;
    opt_.Reset();

    InitSweeps();
    std::uniform_int_distribution<int> distribution(0,trainSamples_.rows()-1);

    for (int i = 0; i < niter_opt_; i++) {
      //Sample();
      int index;
      batch_samples.resize(batchsize_node_,psi_.Nvisible());
      for(int k=0;k<batchsize_node_;k++){
        index = distribution(rgen_);
        batch_samples.row(k) = trainSamples_.row(index);
      }

      Gradient(batch_samples);
      
      UpdateParameters();
      Scan(i);
      
      PrintOutput(i);
    }
  }

  void UpdateParameters() {
    auto pars = psi_.GetParameters();

    if (dosr_) {
      const int nsamp = vsamp_.rows();

      Eigen::VectorXcd b = Ok_.adjoint() * elocs_;
      SumOnNodes(b);
      b /= double(nsamp * totalnodes_);

      if (!use_iterative_) {
        // Explicit construction of the S matrix
        Eigen::MatrixXcd S = Ok_.adjoint() * Ok_;
        SumOnNodes(S);
        S /= double(nsamp * totalnodes_);

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
        S.setShift(sr_diag_shift_);
        S.setScale(1. / double(nsamp * totalnodes_));

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

  void PrintOutput(int i) {
    // Note: This has to be called in all MPI processes, because converting the
    // ObsManager to JSON performs a MPI reduction.
    auto obs_data = json(obsmanager_);
    if (output_.has_value()) {  // output_.has_value() iff the MPI rank is 0, so
                                // the output is only written once
      output_->WriteLog(i, obs_data);
      output_->WriteState(i, psi_);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void setSrParameters(double diagshift = 0.01, bool rescale_shift = false,
                       bool use_iterative = false) {
    sr_diag_shift_ = diagshift;
    sr_rescale_shift_ = rescale_shift;
    use_iterative_ = use_iterative;
    dosr_ = true;
  }
 

  //Compute the partition function by exact enumeration 
  void ExactPartitionFunction() {
      Z_ = 0.0;
      for(int i=0;i<basis_states_.rows();i++){
          Z_ += std::norm(std::exp(psi_.LogVal(basis_states_.row(i))));
      }
  }

  // Compute the overlap with the target wavefunction
  void Fidelity(){
      std::complex<double> tmp;
      for(int i=0;i<basis_states_.rows();i++){
          tmp += wf_(i)*std::abs(std::exp(psi_.LogVal(basis_states_.row(i))))/std::sqrt(Z_);
      }
      fidelity_ = std::norm(tmp);
  }
  
  //Compute KL divergence exactly
  void ExactKL(){
    //KL in the standard basis
    KL_ = 0.0;
    //complex<double> tmp;
    for(int i=0;i<basis_states_.rows();i++){
      if (std::norm(wf_(i))>0.0){
        KL_ += std::norm(wf_(i))*log(std::norm(wf_(i)));
      }
      KL_ -= std::norm(wf_(i))*log(std::norm(exp(psi_.LogVal(basis_states_.row(i)))));
      KL_ += std::norm(wf_(i))*log(Z_);
    }
  }

  // Test the derivatives of the KL divergence
  void TestDerKL(double eps=0.00001){
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
      alg_ders -= 2.0*std::norm(wf_(j))*psi_.DerLog(basis_states_.row(j));
      alg_ders += 2.0*(std::norm(std::exp(psi_.LogVal(basis_states_.row(j))))/Z_) * psi_.DerLog(basis_states_.row(j));
      //alg_ders +=  2.0*std::norm(wf_(j))*psi_.DerLog(basis_states_.row(j)).real();
      //alg_ders -= 2.0*(std::norm(std::exp(psi_.LogVal(basis_states_.row(j))))/Z_) *psi_.DerLog(basis_states_.row(j)).real();
    }
      
    //-- NUMERICAL DERIVATIVES --//
//    std::cout<<"\n- - - - - - - - - - - -\n"<< "Real Part of Derivatives"<<std::endl;
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
      num_ders_imag(p)=(-valm+valp)/(I_*eps*2.0);
      std::cout<<"Numerical Gradient = (";
      std::cout<<num_ders_real(p).real()<<" , "<<num_ders_imag(p).imag()<<")\t-->";
      std::cout<<"(";
      std::cout<<alg_ders(p).real()<<" , "<<alg_ders(p).imag()<<")     ";
      //std::cout<<std::setprecision(8)<<num_ders(p).real() <<"\t\t"<<std::setprecision(8)<<alg_ders(p).real()<<"\t\t\t\t";
      std::cout<<std::endl; 
    }
  }

  void LoadTrainingData(){
    int trainSize = 10000;
    trainSamples_.resize(trainSize,psi_.Nvisible());
    std::string fileName = "data_tfim10.txt";
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
    std::string fileName = "wf_tfim10.txt";
    std::ifstream fin(fileName);
    wf_.resize(1<<psi_.Nvisible());
    for(int i=0;i<1<<psi_.Nvisible();i++){
      fin >> wf_(i);
    }
  }

  //Compute different estimators for the training performance
  void Scan(int i){//,Eigen::MatrixXd &nll_test,std::ofstream &obs_out){
    ExactPartitionFunction();
    ExactKL(); 
    Fidelity();
    PrintStats(i);
  }
  
  //Print observer
  void PrintStats(int i){
      std::cout << "Epoch: " << i << " \t";     
      std::cout << "KL = " << std::setprecision(10) << KL_ << " \t";
      std::cout << "Fidelity = " << std::setprecision(10) << fidelity_<< "\t";//<< Fcheck_;
      std::cout << std::endl;
  } 
 
};

}  // namespace netket

#endif  // NETKET_UNSUPERVISED_HPP
