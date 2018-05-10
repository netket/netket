CXX	=	mpicxx

EIGEN_INCLUDE=External/

#Optimized running flags
CXXFLAGS	= -Ofast -DNDEBUG -I $(EIGEN_INCLUDE)  -I . -std=c++11 -Wall


#Debug-mode flags
# CXXFLAGS =     -O2 -I $(EIGEN_INCLUDE) -std=c++11 -Wall


netket :
	$(CXX) netket.cc $(CXXFLAGS) $(LFLAGS) -o netket


clean	:	cleano cleant cleanout cleanlog

cleano	:
	rm -f netket

cleant	:
	rm -f *.*~

cleanout	:
	rm -f *.out

cleanlog	:
	rm -f *.log
