CXX	=	mpicxx

EIGEN_INCLUDE=../../External/

#Optimized running flags
CXXFLAGS	= -Ofast -DNDEBUG -I $(EIGEN_INCLUDE)  -std=c++11 -I ../../


#Debug-mode flags
# CXXFLAGS =     -O2 -I $(EIGEN_INCLUDE) -std=c++11 -I ../../



stats :
	$(CXX) stats.cc $(CXXFLAGS) $(LFLAGS) -o stats.o


vector_stats:
	$(CXX) vector_stats.cc $(CXXFLAGS) $(LFLAGS) -o vector_stats.o


obsmanager:
	$(CXX) obsmanager.cc $(CXXFLAGS) $(LFLAGS) -o obsmanager.o

clean	:	cleano cleant cleanout cleanlog

cleano	:
	rm -f *.o

cleant	:
	rm -f *.*~

cleanout	:
	rm -f *.out

cleanlog	:
	rm -f *.log
