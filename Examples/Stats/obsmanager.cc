#include <iostream>
#include <random>
#include <vector>
#include <random>
#include <cassert>
#include "netket.hpp"

using namespace std;
using namespace netket;

double loggaussian(double x){
  return -(x*x);
}

double GaussianWalk(double x,std::mt19937 & gen,double step_size){
  std::uniform_real_distribution<> dis(0, 1.0);

  double xnew=x+(dis(gen)-0.5)*step_size;

  if(std::exp(loggaussian(xnew)-loggaussian(x))>dis(gen)){
    return xnew;
  }
  else{
    return x;
  }
}

int main(int argc,char * argv[]){
  MPI_Init(&argc,&argv);

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()

  double x=0;

  ObsManager obm;

  obm.AddObservable("GaussianWalk1",double());

  for(int i=0;i<100000;i++){
    x=GaussianWalk(x,gen,1);
    obm.Push("GaussianWalk1",x);
  }

  obm.AddObservable("GaussianWalk2",double());
  for(int i=0;i<100000;i++){
    x=GaussianWalk(x,gen,2);
    obm.Push("GaussianWalk2",x);
  }

  obm.AddObservable("VectorObs",VectorXd());
  VectorXd gw(2);
  gw.setZero();

  for(int i=0;i<100000;i++){
    for(int k=0;k<2;k++){
      gw(k)=GaussianWalk(gw(k),gen,k+1);
    }
    obm.Push("VectorObs",gw);
  }


  // cout<<json(obm)<<endl;

  json j;
  auto jout=json(obm);
  jout["Iteration"]=1;
  j["Output"].push_back(jout);

  jout=json(obm);
  jout["Iteration"]=2;
  j["Output"].push_back(jout);
  cout<<j<<endl;

  MPI_Finalize();

}
