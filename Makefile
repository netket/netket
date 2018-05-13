CXX	=	mpicxx

EXTERNAL_INCLUDE=External/

#Optimized running flags
CXXFLAGS	= -Ofast -DNDEBUG -I $(EXTERNAL_INCLUDE)  -I . -std=c++11 -Wall -Wextra -Wshadow -pedantic


#Debug-mode flags
#CXXFLAGS =     -O2 -I $(EXTERNAL_INCLUDE) -I . -std=c++11 -Wall -Wextra -Wshadow -pedantic


netket :
	$(CXX) netket.cc $(CXXFLAGS) -o netket


clean	:	cleano cleant cleanout cleanlog

cleano	:
	rm -f netket

cleant	:
	rm -f *.*~

cleanout	:
	rm -f *.out

cleanlog	:
	rm -f *.log
