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

#ifndef NETKET_ABSTRACTLEARNING_HH
#define NETKET_ABSTRACTLEARNING_HH

#include <vector>

namespace netket{

template<class Ham,class Psi,class Samp,class Opt> class AbstractLearning{
public:
  virtual void Sample(double nsweeps)=0;
  virtual void SetOutName(string filebase, double freq=100)=0;
  virtual void Gradient()=0;
  virtual std::complex<double> Eloc(const VectorXd & v)=0;
  virtual double ElocMean()=0;
  virtual double Elocvar()=0;
  virtual void Run(double nsweeps,double niter)=0;
  virtual void UpdateParameters()=0;
  virtual void PrintOutput(double i)=0;
  virtual void CheckDerLog(double eps=1.0e-4)=0;
};

}
#endif
