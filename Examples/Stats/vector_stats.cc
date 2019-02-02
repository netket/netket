#include <iostream>
#include <random>
#include <vector>
#include <random>
#include <cassert>
#include <valarray>

#include "netket.hpp"

using namespace std;
using namespace netket;

double loggaussian(double x,double mean){
  return -((x-mean)*(x-mean));
}

double GaussianWalk(double x,std::mt19937 & gen,double mean,double step_size){
  std::uniform_real_distribution<> dis(0, 1.0);

  double xnew=x+(dis(gen)-0.5)*step_size;

  if(std::exp(loggaussian(xnew,mean)-loggaussian(x,mean))>dis(gen)){
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

  Binning<VectorXd> binning(16);

  VectorXd x(10);
  x.setZero();

  for(int i=0;i<100000;i++){
    for(int j=0;j<10;j++){
      x(j)=GaussianWalk(x(j),gen,j,0.2*(1+j));
    }
    binning<<x;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto taucorr=binning.TauCorr();
  auto mean=binning.Mean();
  auto eom=binning.ErrorOfMean();
  int N=binning.N();

  if(rank==0){
    cout<<N<<" "<<endl;
    for(int i=0;i<mean.size();i++){
      cout<<mean(i)<<"  "<<eom(i)<<"  "<<taucorr(i)<<endl;
    }
  }

  json j=binning.AllStats();
  cout<<j<<endl;
  MPI_Finalize();

}
