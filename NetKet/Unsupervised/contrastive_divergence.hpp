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

  Eigen::MatrixXd trainSamples_;
  Eigen::VectorXd wf_;

  int batchsize_=1000; //TODO
  int batchsize_node_;
  int cd_;
  int nsamples_;
  int nsamples_node_;
  int ninitsamples_;
  int ndiscardedsamples_;
  int niter_opt_;

  std::complex<double> elocmean_;
  double elocvar_;
  int npar_;
  
  netket::default_random_engine rgen_;
  public:

  ContrastiveDivergence(Hamiltonian &ham, Sampler<Machine<GsType>> &sampler,
                        Optimizer &opt, const json &pars)
      : ham_(ham),
        sampler_(sampler),
        psi_(sampler.Psi()),
        opt_(opt),
        obs_(Observable::FromJson(ham.GetHilbert(), pars)),
        elocvar_(0.) {
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
    
    //TODO Remove dummy loading functions
    LoadTrainingData();
    LoadWavefunction();
    
    npar_ = psi_.Npar();

    opt_.Init(psi_.GetParameters());

    grad_.resize(npar_);
    Okmean_.resize(npar_);

    setSrParameters();

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    //batch_size_ = ...
    batchsize_node_ = int(std::ceil(double(batchsize_) / double(totalnodes_)));

    nsamples_ = FieldVal(pars["Unsupervised"], "Nsamples", "GroundState");

    nsamples_node_ = int(std::ceil(double(nsamples_) / double(totalnodes_)));

    ninitsamples_ =
        FieldOrDefaultVal(pars["Unsupervised"], "DiscardedSamplesOnInit", 0.);

    ndiscardedsamples_ = FieldOrDefaultVal(
        pars["Unsupervised"], "DiscardedSamples", 0.1 * nsamples_node_);

    niter_opt_ = FieldVal(pars["Unsupervised"], "NiterOpt", "GroundState");

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

  void Sample() {
    sampler_.Reset();
    vsamp_.resize(nsamples_node_, psi_.Nvisible());
    int index;
    
    //TODO Check the initialization of the distribution
    std::uniform_int_distribution<int> distribution(0,trainSamples_.rows()-1);
    index = distribution(rgen_);
    sampler_.SetVisible(trainSamples_.row(index));
    
    for (int i = 0; i < nsamples_node_; i++) {
      //TODO Check this loop
      for (int j=0; j<cd_*psi_.Nvisible(); j++){
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
    const int ndata = batchSamples.rows();
    Ok_.resize(ndata, psi_.Npar());
    for (int i = 0; i < ndata; i++) {
      Ok_.row(i) = psi_.DerLog(batchSamples.row(i));
    }
    Okmean_ = Ok_.colwise().mean();
    SumOnNodes(Okmean_);
    Okmean_ /= double(totalnodes_);

    grad_ = Ok_;
    SumOnNodes(grad_);
    grad_ /= double(totalnodes_ * ndata);

    //// Negative phase driven by the machine
    //const int nsamp = vsamp_.rows();
    //elocs_.resize(nsamp);
    //Ok_.resize(nsamp, psi_.Npar());

    //for (int i = 0; i < nsamp; i++) {
    //  elocs_(i) = Eloc(vsamp_.row(i));
    //  Ok_.row(i) = psi_.DerLog(vsamp_.row(i));
    //  obsmanager_.Push("Energy", elocs_(i).real());

    //  for (const auto &ob : obs_) {
    //    obsmanager_.Push(ob.Name(), ObSamp(ob, vsamp_.row(i)));
    //  }
    //}

    //elocmean_ = elocs_.mean();
    //SumOnNodes(elocmean_);
    //elocmean_ /= double(totalnodes_);

    //Okmean_ = Ok_.colwise().mean();
    //SumOnNodes(Okmean_);
    //Okmean_ /= double(totalnodes_);

    ////Ok_ = Ok_.rowwise() - Okmean_.transpose();
    //elocs_ -= elocmean_ * Eigen::VectorXd::Ones(nsamp);

    //for (int i = 0; i < nsamp; i++) {
    //  obsmanager_.Push("EnergyVariance", std::norm(elocs_(i)));
    //}

    //grad_ -= Ok_;
    ////grad_ = 2. * (Ok_.adjoint() * elocs_);

    //// Summing the gradient over the nodes
    //SumOnNodes(grad_);
    //grad_ /= double(totalnodes_ * nsamp);
  }

  std::complex<double> Eloc(const Eigen::VectorXd &v) {
    ham_.FindConn(v, mel_, connectors_, newconfs_);

    assert(connectors_.size() == mel_.size());

    auto logvaldiffs = (psi_.LogValDiff(v, connectors_, newconfs_));

    assert(mel_.size() == std::size_t(logvaldiffs.size()));

    std::complex<double> eloc = 0;

    for (int i = 0; i < logvaldiffs.size(); i++) {
      eloc += mel_[i] * std::exp(logvaldiffs(i));
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
      obval += mel_[i] * std::exp(logvaldiffs(i));
    }

    return obval.real();
  }

  double ElocMean() { return elocmean_.real(); }

  double Elocvar() { return elocvar_; }

  void Run(const Eigen::MatrixXd & trainSamples) {
    opt_.Reset();

    InitSweeps();

    for (int i = 0; i < niter_opt_; i++) {
      Sample();

      //Gradient();

      //UpdateParameters();

      //PrintOutput(i);
    }
  }

//  void UpdateParameters() {
//    auto pars = psi_.GetParameters();
//
//    if (dosr_) {
//      const int nsamp = vsamp_.rows();
//
//      Eigen::VectorXcd b = Ok_.adjoint() * elocs_;
//      SumOnNodes(b);
//      b /= double(nsamp * totalnodes_);
//
//      if (!use_iterative_) {
//        // Explicit construction of the S matrix
//        Eigen::MatrixXcd S = Ok_.adjoint() * Ok_;
//        SumOnNodes(S);
//        S /= double(nsamp * totalnodes_);
//
//        // Adding diagonal shift
//        S += Eigen::MatrixXd::Identity(pars.size(), pars.size()) *
//             sr_diag_shift_;
//
//        Eigen::FullPivHouseholderQR<Eigen::MatrixXcd> qr(S.rows(), S.cols());
//        qr.setThreshold(1.0e-6);
//        qr.compute(S);
//        const Eigen::VectorXcd deltaP = qr.solve(b);
//        // Eigen::VectorXcd deltaP=S.jacobiSvd(ComputeThinU |
//        // ComputeThinV).solve(b);
//
//        assert(deltaP.size() == grad_.size());
//        grad_ = deltaP;
//
//        if (sr_rescale_shift_) {
//          std::complex<double> nor = (deltaP.dot(S * deltaP));
//          grad_ /= std::sqrt(nor.real());
//        }
//
//      } else {
//        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper,
//                                 Eigen::IdentityPreconditioner>
//            it_solver;
//        // Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner>
//        // it_solver;
//        it_solver.setTolerance(1.0e-3);
//        MatrixReplacement S;
//        S.attachMatrix(Ok_);
//        S.setShift(sr_diag_shift_);
//        S.setScale(1. / double(nsamp * totalnodes_));
//
//        it_solver.compute(S);
//        auto deltaP = it_solver.solve(b);
//
//        grad_ = deltaP;
//        if (sr_rescale_shift_) {
//          auto nor = deltaP.dot(S * deltaP);
//          grad_ /= std::sqrt(nor.real());
//        }
//
//        // if(mynode_==0){
//        //   std::cerr<<it_solver.iterations()<<"
//        //   "<<it_solver.error()<<std::endl;
//        // }
//        MPI_Barrier(MPI_COMM_WORLD);
//      }
//    }
//
//    opt_.Update(grad_, pars);
//
//    SendToAll(pars);
//
//    psi_.SetParameters(pars);
//    MPI_Barrier(MPI_COMM_WORLD);
//  }

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
 

  void LoadTrainingData(){
    int trainSize = 10000;
    trainSamples_.resize(trainSize,psi_.Nvisible());
    std::string fileName = "data_tfim10.txt";
    std::ifstream fin_samples(fileName);
    for (int n=0; n<trainSize; n++) {
      for (int j=0; j<psi_.Nvisible(); j++) {
        fin_samples>> trainSamples_(n,j);
      }
      //std::cout<<trainSamples_.row(n)<<std::endl;
    }
  }

  void LoadWavefunction(){
    std::string fileName = "wf_tfim10.txt";
    std::ifstream fin(fileName);
    wf_.resize(1<<10);
    for(int i=0;i<1<<10;i++){
      fin >> wf_(i);
      //std::cout<<wf_(i)<<"   ";
    }
  }
  // Setup the training batch and visible layer initial configuration
  void SetUpTrainingStep(Eigen::MatrixXd &batch_samples,
                         std::uniform_int_distribution<int> & distribution){
    int index;
    // Initialize the visible layer to random data samples
    batch_samples.resize(batchsize_node_,psi_.Nvisible());
    for(int k=0;k<batchsize_node_;k++){
      index = distribution(rgen_);
      batch_samples.row(k) = trainSamples_.row(index);
    }
    //NNstate_.SetVisibleLayer(batch_samples);
    //sampler_.SetVisible( 
    //// Build the batch of data
    //batch_samples.resize(batchsize__,psi_.Nvisible()); 
    //for(int k=0;k<batchsize_;k++){
    //  index = distribution(rgen_);
    //  batch_samples.row(k) = trainData.row(index);
    //}
  }

 
};

}  // namespace netket

#endif  // NETKET_UNSUPERVISED_HPP
