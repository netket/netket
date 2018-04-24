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

#ifndef NETKET_MPIINTERF_HH
#define NETKET_MPIINTERF_HH

#include <mpi.h>
#include <vector>
#include <valarray>
#include <complex>
#include <Eigen/Dense>
#include <cassert>

namespace netket{

using namespace std;
using namespace Eigen;

inline void SendToAll(double & val,int sendnode=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(&val,1,MPI_DOUBLE,sendnode,comm);
}
inline void SendToAll(int & val,int sendnode=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(&val,1,MPI_INT,sendnode,comm);
}
inline void SendToAll(complex<double> & val,int sendnode=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(&val,1,MPI_DOUBLE_COMPLEX,sendnode,comm);
}

void SendToAll(vector<int> & value,int root=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(&value[0],value.size(),MPI_INT,root,comm);
}
void SendToAll(vector<double> & value,int root=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(&value[0],value.size(),MPI_DOUBLE,root,comm);
}
void SendToAll(vector<complex<double> > & value,int root=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(&value[0],value.size(),MPI_DOUBLE_COMPLEX,root,comm);
}
void SendToAll(VectorXi & value,int root=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(value.data(),value.size(),MPI_INT,root,comm);
}
void SendToAll(VectorXd & value,int root=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(value.data(),value.size(),MPI_DOUBLE,root,comm);
}
void SendToAll(VectorXcd & value,int root=0,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Bcast(value.data(),value.size(),MPI_DOUBLE_COMPLEX,root,comm);
}

//Accumulates the sum of val collected from all nodes and the sum is distributed back to all processors
inline void SumOnNodes(double & val,double & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(&val,&sum,1,MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(int & val,int & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(&val,&sum,1,MPI_INT,MPI_SUM,comm);
}
inline void SumOnNodes(int & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,&value,1,MPI_INT,MPI_SUM,comm);
}

inline void SumOnNodes(complex<double> & val,complex<double> & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(&val,&sum,1,MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(vector<double> & val,vector<double> & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(&val[0],&sum[0],val.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(vector<double> & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,&value[0],value.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(valarray<double> & val,valarray<double> & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(&val[0],&sum[0],val.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(valarray<double> & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,&value[0],value.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(VectorXd & val,VectorXd & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  assert(sum.size()>=val.size());
  MPI_Allreduce(val.data(),sum.data(),val.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(VectorXd & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,value.data(),value.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(MatrixXd & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,value.data(),value.size(),MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(MatrixXcd & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,value.data(),value.size(),MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(double & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,&value,1,MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(double* value,int size,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,value,size,MPI_DOUBLE,MPI_SUM,comm);
}

inline void SumOnNodes(complex<double> *value,int size,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,value,size,MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(vector<complex<double> > & val,vector<complex<double> > & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(&val[0],&sum[0],val.size(),MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(vector<complex<double> > & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,&value[0],value.size(),MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(complex<double> & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,&value,1,MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(VectorXcd & value,const MPI_Comm comm=MPI_COMM_WORLD){
  MPI_Allreduce(MPI_IN_PLACE,value.data(),value.size(),MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

inline void SumOnNodes(VectorXcd & val,VectorXcd & sum,const MPI_Comm comm=MPI_COMM_WORLD){
  assert(sum.size()>=val.size());
  MPI_Allreduce(val.data(),sum.data(),val.size(),MPI_DOUBLE_COMPLEX,MPI_SUM,comm);
}

}

#endif
