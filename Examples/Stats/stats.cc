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

  Binning<double> binning(16);

  double x=0;

  for(int i=0;i<100000;i++){
    x=GaussianWalk(x,gen,2);
    binning<<x;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double taucorr=binning.TauCorr();
  double mean=binning.Mean();
  double eom=binning.ErrorOfMean();
  int N=binning.N();

  if(rank==0){
    cout<<N<<" "<<mean<<" "<<eom<<" "<<taucorr<<endl;
  }

  if(rank==0){
    cout<<"Resetting"<<endl;
  }
  binning.Reset();

  taucorr=binning.TauCorr();
  mean=binning.Mean();
  eom=binning.ErrorOfMean();
  N=binning.N();

  if(rank==0){
    cout<<N<<" "<<mean<<" "<<eom<<" "<<taucorr<<endl;
  }

  for(int i=0;i<100000;i++){
    x=GaussianWalk(x,gen,2);
    binning<<x;
  }

  taucorr=binning.TauCorr();
  mean=binning.Mean();
  eom=binning.ErrorOfMean();
  N=binning.N();

  if(rank==0){
    cout<<N<<" "<<mean<<" "<<eom<<" "<<taucorr<<endl;
  }

  json j=binning.AllStats();
  cout<<j<<endl;

  MPI_Finalize();

}
