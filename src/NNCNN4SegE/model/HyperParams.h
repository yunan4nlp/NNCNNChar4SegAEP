#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	int wordHiddenSize;
	int wordContext;
	int wordWindow;
	int windowOutput;

	int charHiddenSize;
	int charContext;
	int charWindow;
	int charWindowOutput;

	int evalCharHiddenSize;
	int evalCharContext;
	int evalCharWindow;
	dtype dropProb;


	//auto generated
	int wordDim;
	int charDim;
	int evalCharDim;
	int inputSize;
	int labelSize;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		wordHiddenSize = opt.wordHiddenSize;
		wordContext = opt.wordcontext;

		charHiddenSize = opt.charHiddenSize;
		charContext = opt.charcontext;

		evalCharHiddenSize = opt.evalCharHiddenSize;
		evalCharContext = opt.evalCharContext;
		dropProb = opt.dropProb;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}

public:

	void print(){
	}


	void saveModel(std::ofstream &os) const {
		os << nnRegular << endl;
		os << adaAlpha << endl;
		os << adaEps << endl;

		os << wordHiddenSize << endl;
		os << wordContext << endl;
		os << wordWindow << endl;
		os << windowOutput << endl;

		os << charHiddenSize << endl;
		os << charContext << endl;
		os << charWindow << endl;
		os << charWindowOutput << endl;

		os << evalCharHiddenSize << endl;
		os << evalCharContext << endl;
		os << evalCharWindow << endl;
		os << dropProb << endl;


		os << wordDim << endl;
		os << charDim << endl;
		os << evalCharDim << endl;
		os << inputSize << endl;
		os << labelSize << endl;
	}

	void loadModel(std::ifstream &is) {
		is >> nnRegular;
		is >> adaAlpha;
		is >> adaEps;

		is >> wordHiddenSize;
		is >> wordContext;
		is >> wordWindow;
		is >> windowOutput;

		is >> charHiddenSize;
		is >> charContext;
		is >> charWindow;
		is >> charWindowOutput;

		is >> evalCharHiddenSize;
		is >> evalCharContext;
		is >> evalCharWindow;
		is >> dropProb;


		is >> wordDim;
		is >> charDim;
		is >> evalCharDim;
		is >> inputSize;
		is >> labelSize;
		bAssigned = true;
	}
private:
	bool bAssigned;


};


#endif /* SRC_HyperParams_H_ */