#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Instance.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;
	const static int max_char_length = 4096;

	const static int MAX_EVAL_SIZE = 10;
	const static int MAX_EVAL_LENGTH = 32;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;
	vector<UniNode> _word_hidden;

	AvgPoolNode _word_avg_pooling;
	MaxPoolNode _word_max_pooling;
	MinPoolNode _word_min_pooling;

	ConcatNode _word_pooling_concat;


	vector<LookupNode> _char_inputs;
	WindowBuilder _char_window;
	vector<UniNode> _char_hidden;

	AvgPoolNode _char_avg_pooling;
	MaxPoolNode _char_max_pooling;
	MinPoolNode _char_min_pooling;

	ConcatNode _char_pooling_concat;

	vector<vector<LookupNode> > _eval_char_inputs;
	vector<WindowBuilder> _eval_char_windows;
	vector<vector<UniNode> > _eval_char_hiddens;
	vector<MaxPoolNode> _eval_char_max_poolings;
	vector<MinPoolNode> _eval_char_min_poolings;
	vector<AvgPoolNode> _eval_char_avg_poolings;
	vector<ConcatNode> _eval_char_pooling_concats;
	MaxPoolNode _eval_concat_max_pooling;
	MinPoolNode _eval_concat_min_pooling;
	AvgPoolNode _eval_concat_avg_pooling;

	ConcatNode _eval_pooling_concat;
	Node _eval_concat_bucket;

	ConcatNode _concat;
	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int char_length, int eval_size, int eval_length){
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_word_hidden.resize(sent_length);
		_word_avg_pooling.setParam(sent_length);
		_word_max_pooling.setParam(sent_length);
		_word_min_pooling.setParam(sent_length);

		_char_inputs.resize(char_length);
		_char_window.resize(char_length);
		_char_hidden.resize(char_length);
		_char_avg_pooling.setParam(char_length);
		_char_max_pooling.setParam(char_length);
		_char_min_pooling.setParam(char_length);

		_eval_char_inputs.resize(eval_size);
		_eval_char_windows.resize(eval_size);
		_eval_char_hiddens.resize(eval_size);
		_eval_char_max_poolings.resize(eval_size);
		_eval_char_min_poolings.resize(eval_size);
		_eval_char_avg_poolings.resize(eval_size);
		_eval_char_pooling_concats.resize(eval_size);
		for (int idx = 0; idx < eval_size; idx++) {
			_eval_char_inputs[idx].resize(eval_length);
			_eval_char_windows[idx].resize(eval_length);
			_eval_char_hiddens[idx].resize(eval_length);
			_eval_char_max_poolings[idx].setParam(eval_length);
			_eval_char_min_poolings[idx].setParam(eval_length);
			_eval_char_avg_poolings[idx].setParam(eval_length);
		}

		_eval_concat_max_pooling.setParam(eval_size);
		_eval_concat_min_pooling.setParam(eval_size);
		_eval_concat_avg_pooling.setParam(eval_size);

	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_word_window.clear();
		_word_hidden.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
			_word_hidden[idx].setParam(&model.hidden_linear);
			_word_hidden[idx].init(opts.wordHiddenSize, opts.dropProb, mem);
		}
		_word_window.init(opts.wordDim, opts.wordContext, mem);
		_word_avg_pooling.init(opts.wordHiddenSize, -1, mem);
		_word_max_pooling.init(opts.wordHiddenSize, -1, mem);
		_word_min_pooling.init(opts.wordHiddenSize, -1, mem);
		_word_pooling_concat.init(opts.wordHiddenSize * 3, -1, mem);

		for (int idx = 0; idx < _char_inputs.size(); idx++) {
			_char_inputs[idx].setParam(&model.chars);
			_char_inputs[idx].init(opts.charDim, opts.dropProb, mem);
			_char_hidden[idx].setParam(&model.char_hidden_linear);
			_char_hidden[idx].init(opts.charHiddenSize, opts.dropProb, mem);
		}
		_char_window.init(opts.charDim, opts.charContext, mem);
		_char_avg_pooling.init(opts.charHiddenSize, -1, mem);
		_char_max_pooling.init(opts.charHiddenSize, -1, mem);
		_char_min_pooling.init(opts.charHiddenSize, -1, mem);
		_char_pooling_concat.init(opts.charHiddenSize * 3, -1, mem);

		int max_eval_size = _eval_char_inputs.size();
		for (int idx = 0; idx < max_eval_size; idx++) {
			int max_eval_char_size = _eval_char_inputs[idx].size();
			for (int idy = 0; idy < max_eval_char_size; idy++) {
				_eval_char_inputs[idx][idy].setParam(&model.evalChars);
				_eval_char_inputs[idx][idy].init(opts.evalCharDim, opts.dropProb, mem);
				_eval_char_hiddens[idx][idy].setParam(&model.eval_char_hidden_linear);
				_eval_char_hiddens[idx][idy].init(opts.evalCharHiddenSize, opts.dropProb, mem);
			}
			_eval_char_windows[idx].init(opts.evalCharDim, opts.evalCharContext, mem);
			_eval_char_max_poolings[idx].init(opts.evalCharHiddenSize, -1, mem);
			_eval_char_min_poolings[idx].init(opts.evalCharHiddenSize, -1, mem);
			_eval_char_avg_poolings[idx].init(opts.evalCharHiddenSize, -1, mem);
			_eval_char_pooling_concats[idx].init(opts.evalCharHiddenSize * 3, -1, mem);
		}

		_eval_concat_max_pooling.init(opts.evalCharHiddenSize * 3, -1, mem);
		_eval_concat_min_pooling.init(opts.evalCharHiddenSize * 3, -1, mem);
		_eval_concat_avg_pooling.init(opts.evalCharHiddenSize * 3, -1, mem);

		_eval_pooling_concat.init(opts.evalCharHiddenSize * 3 * 3, -1, mem);
		_eval_concat_bucket.init(opts.evalCharHiddenSize * 3 * 3, -1, mem);
		_eval_concat_bucket.set_bucket();

		_concat.init(opts.charHiddenSize * 3 + opts.wordHiddenSize * 3 + opts.evalCharHiddenSize * 3 * 3, -1, mem);
		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Instance& inst, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		int words_num = inst.m_segs.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int i = 0; i < words_num; i++) {
			_word_inputs[i].forward(this, inst.m_segs[i]);
		}
		_word_window.forward(this, getPNodes(_word_inputs, words_num));

		for (int i = 0; i < words_num; i++) {
			_word_hidden[i].forward(this, &_word_window._outputs[i]);
		}
		_word_max_pooling.forward(this, getPNodes(_word_hidden, words_num));
		_word_min_pooling.forward(this, getPNodes(_word_hidden, words_num));
		_word_avg_pooling.forward(this, getPNodes(_word_hidden, words_num));
		_word_pooling_concat.forward(this, &_word_max_pooling, &_word_min_pooling, &_word_avg_pooling);

		int chars_num = inst.m_seg_chars.size();
		if (chars_num > max_char_length)
			chars_num = max_char_length;
		for (int i = 0; i < chars_num; i++) {
			_char_inputs[i].forward(this, inst.m_seg_chars[i]);
		}
		_char_window.forward(this, getPNodes(_char_inputs, chars_num));

		for (int i = 0; i < chars_num; i++) {
			_char_hidden[i].forward(this, &_char_window._outputs[i]);
		}
		_char_max_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_char_min_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_char_avg_pooling.forward(this, getPNodes(_char_hidden, chars_num));
		_char_pooling_concat.forward(this, &_char_max_pooling, &_char_min_pooling, &_char_avg_pooling);

		int eval_size = inst.m_eval_chars.size();
		if (eval_size > MAX_EVAL_SIZE)
			eval_size = MAX_EVAL_SIZE;
		int eval_length;
		for (int i = 0; i < eval_size; i++) {
			eval_length = inst.m_eval_chars[i].size();
			if (eval_length > MAX_EVAL_LENGTH)
				eval_length = MAX_EVAL_LENGTH;
			for (int j = 0; j < eval_length; j++) {
				_eval_char_inputs[i][j].forward(this, inst.m_eval_chars[i][j]);
			}
			_eval_char_windows[i].forward(this, getPNodes(_eval_char_inputs[i], eval_length));
			for (int j = 0; j < eval_length; j++) {
				_eval_char_hiddens[i][j].forward(this, &_eval_char_windows[i]._outputs[j]);
			}
			_eval_char_max_poolings[i].forward(this, getPNodes(_eval_char_hiddens[i], eval_length));
			_eval_char_min_poolings[i].forward(this, getPNodes(_eval_char_hiddens[i], eval_length));
			_eval_char_avg_poolings[i].forward(this, getPNodes(_eval_char_hiddens[i], eval_length));
			_eval_char_pooling_concats[i].forward(this, &_eval_char_max_poolings[i], &_eval_char_min_poolings[i], &_eval_char_avg_poolings[i]);
		}
		if (eval_size == 0)
			_concat.forward(this, &_word_pooling_concat, &_char_pooling_concat, &_eval_concat_bucket);
		else {
			_eval_concat_max_pooling.forward(this, getPNodes(_eval_char_pooling_concats, eval_size));
			_eval_concat_min_pooling.forward(this, getPNodes(_eval_char_pooling_concats, eval_size));
			_eval_concat_avg_pooling.forward(this, getPNodes(_eval_char_pooling_concats, eval_size));
			_eval_pooling_concat.forward(this, &_eval_concat_max_pooling, &_eval_concat_min_pooling, &_eval_concat_avg_pooling);
			_concat.forward(this, &_word_pooling_concat, &_char_pooling_concat, &_eval_pooling_concat);
		}
		_output.forward(this, &_concat);
	}
};

#endif /* SRC_ComputionGraph_H_ */