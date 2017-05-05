#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside

	Alphabet charAlpha; // should be initialized outside
	LookupTable chars; // should be initialized outside

	Alphabet sparseAlpha;// should be initialized outside
	SparseParams sparse_linear;
	UniParams hidden_linear;
	UniParams char_hidden_linear;
	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.charDim = chars.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;

		opts.charWindow = opts.charContext * 2 + 1;
		opts.charWindowOutput = opts.charDim * opts.charWindow;

		opts.labelSize = labelAlpha.size();
		hidden_linear.initial(opts.wordHiddenSize, opts.windowOutput, true, mem);
		sparse_linear.initial(&sparseAlpha, opts.labelSize);
		char_hidden_linear.initial(opts.charHiddenSize, opts.charWindowOutput, true, mem);
		opts.inputSize = (opts.wordHiddenSize + opts.charHiddenSize) * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		sparse_linear.exportAdaParams(ada);
		hidden_linear.exportAdaParams(ada);
		char_hidden_linear.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&hidden_linear.W, "hidden_linear.W");
		checkgrad.add(&hidden_linear.b, "hidden_linear.b");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */