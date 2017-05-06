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

	LSTMParams rnn_params;
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
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;

		opts.charDim = chars.nDim;
		opts.charWindow = opts.charContext * 2 + 1;
		opts.charWindowOutput = opts.charDim * opts.charWindow;

		opts.labelSize = labelAlpha.size();
		rnn_params.initial(opts.wordHiddenSize, opts.windowOutput, mem);
		char_hidden_linear.initial(opts.charHiddenSize, opts.charWindowOutput, true, mem);
		opts.inputSize = (opts.charHiddenSize + opts.wordHiddenSize ) * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		char_hidden_linear.exportAdaParams(ada);
		rnn_params.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */