#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Instance.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;
	const static int max_char_length = 4096;
	const static int max_att_size = 10;

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

	vector<LookupNode> _att_inputs;
	AvgPoolNode _att_avg_pooling;
	MaxPoolNode _att_max_pooling;
	MinPoolNode _att_min_pooling;
	ConcatNode _att_pooling_concat;
	Node _att_bucket;

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
	inline void createNodes(int sent_length, int char_length, int att_size) {
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

		_att_inputs.resize(att_size);
		_att_avg_pooling.setParam(att_size);
		_att_max_pooling.setParam(att_size);
		_att_min_pooling.setParam(att_size);
	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_att_inputs.clear();
		_word_window.clear();
		_word_hidden.clear();

		_att_inputs.clear();
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

		for (int idx = 0; idx < _att_inputs.size(); idx++) {
			_att_inputs[idx].setParam(&model.atts);
			_att_inputs[idx].init(opts.attDim, opts.dropProb, mem);
		}
		_att_bucket.init(opts.attDim, -1, mem);
		_att_bucket.set_bucket();

		_att_avg_pooling.init(opts.attDim, -1, mem);
		_att_max_pooling.init(opts.attDim, -1, mem);
		_att_min_pooling.init(opts.attDim, -1, mem);
		_att_pooling_concat.init(opts.attDim * 3, -1, mem);

		_concat.init((opts.attDim + opts.wordHiddenSize) * 3, -1, mem);
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
		if (chars_num > max_sentence_length)
			chars_num = max_sentence_length;
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

		int atts_num = inst.m_attributes.size();
		if (atts_num > max_att_size)
			atts_num = max_att_size;
		for (int i = 0; i < atts_num; i++) {
			_att_inputs[i].forward(this, inst.m_attributes[i]);
		}
		if(atts_num == 0)
			_att_pooling_concat.forward(this, &_att_bucket, &_att_bucket, &_att_bucket);
		else
		{
			_att_avg_pooling.forward(this, getPNodes(_att_inputs, atts_num));
			_att_max_pooling.forward(this, getPNodes(_att_inputs, atts_num));
			_att_min_pooling.forward(this, getPNodes(_att_inputs, atts_num));
			_att_pooling_concat.forward(this, &_att_avg_pooling, &_att_max_pooling, &_att_min_pooling);
		}

		_concat.forward(this, &_word_pooling_concat, &_att_pooling_concat);
		_output.forward(this, &_concat);
	}
};

#endif /* SRC_ComputionGraph_H_ */