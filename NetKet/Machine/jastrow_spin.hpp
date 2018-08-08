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
//
//
//

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Lookup/lookup.hpp"
#include "Utils/all_utils.hpp"


#ifndef NETKET_JASTROW_HPP
#define NETKET_JASTROW_HPP






namespace netket {

/** Jastrow machine class.
*
*/
template <typename T>
class Jastrow : public AbstractMachine<T> {
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;


  //number of visible units
  int nv_;

  //number of parameters
  int npar_;

  //weights
  MatrixType W_;

  //buffers
  VectorType thetas_;
  VectorType thetasnew_;




  const Hilbert & hilbert_;

public:
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;

  // constructor
  explicit Jastrow(const Hilbert &hilbert, const json &pars)
      : nv_(hilbert.Size()), hilbert_(hilbert) {
    from_json(pars);
  }






 void Init(){

  W_.resize(nv_,nv_);

  npar_=nv_*(nv_-1)/2;

  thetas_.resize(nv_);
  thetasnew_.resize(nv_);

  InfoMessage() <<"Jastrow WF Initizialized with nvisible = "<<nv_<<" and nparams = "<<npar_<<std::endl;

 }


int Nvisible() const override {
  return nv_;
}


int Npar() const override{
  return npar_;
}

void InitRandomPars(int seed,double sigma) override {

  VectorType par(npar_);

  netket::RandomGaussian(par,seed,sigma);

  SetParameters(par);
}



 VectorType GetParameters() override {

  VectorType pars(npar_);

  int k=0;


  for(int i=0;i<nv_;i++){
    for(int j=i+1;j<nv_;j++){
      pars(k)=W_(i,j);
      k++;
    }
  }

  return pars;
 }

void SetParameters(const VectorType & pars) override {
  int k=0;

  for(int i=0;i<nv_;i++){
    for(int j=i+1;j<nv_;j++){
      W_(i,j)=pars(k);
      W_(j,i)=W_(i,j); //create the lover triangle
      W_(i,i)=T(0);
      k++;
    }
  }
}


void InitLookup(const Eigen::VectorXd & v,LookupType & lt) override {
  if(lt.VectorSize()==0){
    lt.AddVector(v.size());
  }
  if(lt.V(0).size()!=v.size()){
    lt.V(0).resize(v.size());
  }

  lt.V(0)=(W_.transpose()*v); //does not matter the transpose W is symm

  }

// same as for the RBM
void UpdateLookup(const Eigen::VectorXd & v,const std::vector<int>  & tochange,
          const std::vector<double> & newconf,LookupType & lt) override {

  if(tochange.size()!=0){

    for(std::size_t s=0;s<tochange.size();s++){
      const int sf=tochange[s];
      lt.V(0)+=W_.row(sf)*(newconf[s]-v(sf));
    }

  }
}


T LogVal(const Eigen::VectorXd & v) override {
  T logpsi=0;

    for(int i=0;i<nv_;i++){
        for(int j=i+1;j<nv_;j++){
        logpsi+=W_(i,j)*v(i)*v(j);
      }
    }

    return logpsi;
  }


//Value of the logarithm of the wave-function
//using pre-computed look-up tables for efficiency
T LogVal(const Eigen::VectorXd & v, const LookupType & lt) override {

  return 0.5*v.dot(lt.V(0));  //if i use the matrix vector with W i have double counting
}



 //Difference between logarithms of values, when one or more visible variables are being flipped
 VectorType LogValDiff(const Eigen::VectorXd & v,
        const std::vector<std::vector<int> >  & tochange,
        const std::vector<std::vector<double>> & newconf) override {



  const std::size_t nconn=tochange.size();
  VectorType logvaldiffs=VectorType::Zero(nconn);

  thetas_=(W_.transpose()*v);
  T logtsum= 0.5*v.dot(thetas_);

  for(std::size_t k=0;k<nconn;k++){

    if(tochange[k].size()!=0){

      thetasnew_=thetas_;
      Eigen::VectorXd vnew=v;

      for(std::size_t s=0;s<tochange[k].size();s++){
        const int sf=tochange[k][s];

          thetasnew_+=W_.row(sf)*(newconf[k][s]-v(sf));
          vnew[sf]=newconf[k][s];
      }


      logvaldiffs(k) = 0.5*vnew.dot(thetasnew_) - logtsum;

    }
  }
  return logvaldiffs;
}


T LogValDiff(const Eigen::VectorXd & v,const std::vector<int>  & tochange,
                 const std::vector<double> & newconf,const LookupType & lt) override {

  T logvaldiff=0.;

  if(tochange.size()!=0){

      T logtsum= 0.5*v.dot(lt.V(0));
      thetasnew_=lt.V(0);
      Eigen::VectorXd vnew=v;

      for(std::size_t s=0;s<tochange.size();s++){
        const int sf=tochange[s];


        thetasnew_+=W_.row(sf)*(newconf[s]-v(sf));
        vnew[sf]=newconf[s];

      }


      logvaldiff = 0.5*vnew.dot(thetasnew_)-logtsum;
      }

    return logvaldiff;
  }




VectorType DerLog(const Eigen::VectorXd & v) override {
  VectorType der(npar_);

  int k=0;


  for(int i=0;i<nv_;i++){
    for(int j=i+1;j<nv_;j++){
      der(k)=v(i)*v(j);
      k++;
    }
  }

  return der;
}


void to_json(json &j)const override {
  j["Machine"]["Name"]="Jastrow";
  j["Machine"]["Nvisible"]=nv_;
  j["Machine"]["W"]=W_;


}

void from_json(const json & pars) override {

  if(pars.at("Machine").at("Name")!="Jastrow"){
    throw InvalidInputError(
          "Error while constructing Jastrow from Json input");
  }

  if (FieldExists(pars["Machine"], "Nvisible")) {
    nv_ = pars["Machine"]["Nvisible"];
  }
  if (nv_ != hilbert_.Size()) {
    throw InvalidInputError(
      "Number of visible units is incompatible with given "
      "Hilbert space");
  }



  Init();


  if( FieldExists(pars["Machine"],"W")){
    W_=pars["Machine"]["W"];
  }
}




};
} // namespace netket

#endif
