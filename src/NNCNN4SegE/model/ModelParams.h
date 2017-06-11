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

	Alphabet evalCharAlpha;// should be initialized outside
	LookupTable evalChars;// should be initialized outside
	UniParams eval_char_hidden_linear;
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
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.windowOutput = opts.wordDim * opts.wordWindow;

		opts.charDim = chars.nDim;
		opts.charWindow = opts.charContext * 2 + 1;
		opts.charWindowOutput = opts.charDim * opts.charWindow;

		opts.evalCharDim = evalChars.nDim;
		opts.evalCharWindow = opts.evalCharContext * 2 + 1;
		opts.labelSize = labelAlpha.size();
		hidden_linear.initial(opts.wordHiddenSize, opts.windowOutput, true, mem);
		char_hidden_linear.initial(opts.charHiddenSize, opts.charWindowOutput, true, mem);

		eval_char_hidden_linear.initial(opts.evalCharHiddenSize, opts.evalCharDim * opts.evalCharWindow, true, mem);
		opts.inputSize = opts.charHiddenSize * 3 + opts.wordHiddenSize * 3 + opts.evalCharHiddenSize * 3 * 3;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}

	bool testInitial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

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

		opts.evalCharDim = evalChars.nDim;
		opts.evalCharWindow = opts.evalCharContext * 2 + 1;
		opts.labelSize = labelAlpha.size();

		opts.inputSize = opts.charHiddenSize * 3 + opts.wordHiddenSize * 3 + opts.evalCharHiddenSize * 3 * 3;
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		hidden_linear.exportAdaParams(ada);
		char_hidden_linear.exportAdaParams(ada);
		evalChars.exportAdaParams(ada);
		eval_char_hidden_linear.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&hidden_linear.W, "hidden_linear.W");
		checkgrad.add(&hidden_linear.b, "hidden_linear.b");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(std::ofstream &os) const {
		wordAlpha.write(os);
		words.save(os);

		charAlpha.write(os);
		chars.save(os);

		evalCharAlpha.write(os);
		evalChars.save(os);
		eval_char_hidden_linear.save(os);
		hidden_linear.save(os);
		char_hidden_linear.save(os);
		olayer_linear.save(os);
		labelAlpha.write(os);
	}

	void loadModel(std::ifstream &is, AlignedMemoryPool* mem = NULL) {
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);

		charAlpha.read(is);
		chars.load(is, &charAlpha, mem);

		evalCharAlpha.read(is);
		evalChars.load(is, &evalCharAlpha, mem);
		eval_char_hidden_linear.load(is);
		hidden_linear.load(is);
		char_hidden_linear.load(is);
		olayer_linear.load(is);
		labelAlpha.read(is);
	}
};

#endif /* SRC_ModelParams_H_ */