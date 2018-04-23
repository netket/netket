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

#ifndef NETKET_LEARNING_CC
#define NETKET_LEARNING_CC

namespace netket{

template<class Ham,class Psi,class Samp,class Opt> class Learning : public AbstractLearning<Ham, Psi, Samp, Opt> {
  AbstractLearning<Ham, Psi, Samp, Opt> * s_;
public:
  Learning(Ham & ham, Samp & sam, Opt & opt, const json & pars){

    if(!FieldExists(pars,"Learning")){
      cerr<<"Learning field is not defined in the input"<<endl;
      std::abort();
    }

    if(!FieldExists(pars["Learning"],"Method")){
      cerr<<"Learning Method is not defined in the input"<<endl;
      std::abort();
    }

    if(pars["Learning"]["Method"]=="Gd" || pars["Learning"]["Method"]=="Sr"){
      s_=new Sr<Ham,Psi,Samp,Opt>(ham,sam,opt,pars);
    }
    else{
      cout<<"Learning method not found"<<endl;
      cout<<pars["Learning"]["Method"]<<endl;
      std::abort();
    }
  }

  void Sample(double nsweeps){
    return s_->Sample(nsweeps);
  }
  void SetOutName(string filebase, double freq=100){
    return s_->SetOutName(filebase,freq);
  }
  void Gradient(){
    return s_->Gradient();
  }
  std::complex<double> Eloc(const VectorXd & v){
    return s_->Eloc(v);
  }
  double ElocMean(){
    return s_->ElocMean();
  }
  double Elocvar(){
    return s_->Elocvar();
  }
  void Run(double nsweeps,double niter){
    return s_->Run(nsweeps,niter);
  }
  void UpdateParameters(){
    return s_->UpdateParameters();
  }
  void PrintOutput(double i){
    return s_->PrintOutput(i);
  }
  void CheckDerLog(double eps=1.0e-4){
    return s_->CheckDerLog(eps);
  }
};
}

#endif
