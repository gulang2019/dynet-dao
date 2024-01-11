/* This is an implementation of Transformer architecture from https://arxiv.org/abs/1706.03762 (Attention is All You need).
* Developed by Cong Duy Vu Hoang
* Updated: 1 Nov 2017
*/
#include <DAO/DAO.h> 
#include <DAO/interface-dynet.h>
#include "transformer-lm.h"
#include "common.h"

using namespace std;
using namespace dynet;
using namespace transformer;

#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <sys/stat.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace dynet;
using namespace transformer;
using namespace boost::program_options;

// hyper-paramaters for training
unsigned PAD_TOKENS = 2; // <SOS> <EOS>
unsigned MINIBATCH_SIZE = 1;

bool DEBUGGING_FLAG = false;

bool PRINT_GRAPHVIZ = false;

unsigned TREPORT = 50;
unsigned DREPORT = 5000;

bool RESET_IF_STUCK = false;
bool SWITCH_TO_ADAM = false;
bool USE_SMALLER_MINIBATCH = false;
unsigned NUM_RESETS = 1;

bool SAMPLING_TRAINING = false;

bool VERBOSE = false;

// ---
bool load_data(const variables_map& vm
	, WordIdSentences& train_cor, WordIdSentences& devel_cor
	, dynet::Dict& d
	, SentinelMarkers& sm);
// ---
bool load_data_gpt2(const variables_map& vm
	, WordIdSentences& train_cor, WordIdSentences& devel_cor, WordIdSentences& test_cor
	, const gpt_vocab& d);
// ---

// ---
bool load_model_config(const std::string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerLModel>>& v_models
	, gpt_vocab& d
	, const transformer::SentinelMarkers& sm);
// ---

// ---
void save_config(const std::string& config_out_file
	, const std::string& params_out_file
	, const TransformerConfig& tfc);
// ---

// ---
void get_dev_stats(const WordIdSentences &devel_cor
	, int unk_id
	, transformer::ModelStats& dstats);
// ---

//---
std::string get_sentence(const WordIdSentence& source, Dict& d);
//---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model);
// ---

// ---
void report_perplexity_score(std::vector<std::shared_ptr<transformer::TransformerLModel>>& v_tf_models, WordIdSentences &test_cor, unsigned minibatch_size);// support ensemble models
// ---

// ---
void run_train(transformer::TransformerLModel &tf, WordIdSentences &train_cor, WordIdSentences &devel_cor, 
	dynet::Trainer* &p_sgd, 
	const std::string& model_path, 
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints);// support batching
// ---

//************************************************************************************************************************************************************
int main(int argc, char** argv) {
	cerr << "*** DyNet initialization ***" << endl;
	auto dyparams = dynet::extract_dynet_params(argc, argv);
	dynet::initialize(dyparams);	

	// command line processing
	variables_map vm; 
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("config,c", value<std::string>(), "config file specifying additional command line options")
		//-----------------------------------------
		("train,t", value<std::vector<std::string>>(), "file containing training sentences, with each line consisting of source ||| target.")		
		("devel,d", value<std::string>(), "file containing development sentences.")
		("test,T", value<std::string>(), "file containing testing sentences.")
		("max-seq-len", value<unsigned>()->default_value(0), "limit the sentence length (either source or target); none by default")
		("vocab", value<std::string>()->default_value(""), "file containing vocabulary file; none by default (will be built from train file)")
		("train-percent", value<unsigned>()->default_value(100), "use <num> percent of sentences in training data; full by default")
		//-----------------------------------------
		("minibatch-size,b", value<unsigned>()->default_value(1), "impose the minibatch size for training (support both GPU and CPU); single batch by default")
		("dynet-autobatch", value<unsigned>()->default_value(0), "impose the auto-batch mode (support both GPU and CPU); no by default")
		//-----------------------------------------
		("sgd-trainer", value<unsigned>()->default_value(0), "use specific SGD trainer (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp; 6: cyclical SGD)")
		("sparse-updates", value<bool>()->default_value(true), "enable/disable sparse update(s) for lookup parameter(s); true by default")
		("grad-clip-threshold", value<float>()->default_value(5.f), "use specific gradient clipping threshold (https://arxiv.org/pdf/1211.5063.pdf); 5 by default")
		//-----------------------------------------
		("model-path,p", value<std::string>()->default_value("."), "all files related to the model will be saved in this folder")
		//-----------------------------------------
		("nlayers", value<unsigned>()->default_value(6), "use <num> layers for stacked decoder layers; 6 by default")
		("num-units,u", value<unsigned>()->default_value(512), "n_embd; 512 by default")
		("num-heads,h", value<unsigned>()->default_value(8), "use <num> for number of heads in multi-head attention mechanism; 4 by default")
		("n-ff-units-factor", value<unsigned>()->default_value(4), "use <num> times of input dim for output dim in feed-forward layer; 4 by default")
		//-----------------------------------------
		("emb-dropout-p", value<float>()->default_value(0.1f), "use dropout for word embeddings; 0.1 by default")
		("sublayer-dropout-p", value<float>()->default_value(0.1f), "attention dropout; 0.1 by default")
		("attention-dropout-p", value<float>()->default_value(0.1f), "prob for skipping attention layer; 0.1 by default")
		("ff-dropout-p", value<float>()->default_value(0.1f), "prob for skipping feed-forward layer; 0.1 by default")
		//-----------------------------------------
		("use-label-smoothing", "use label smoothing for cross entropy; no by default")
		("label-smoothing-weight", value<float>()->default_value(0.1f), "impose label smoothing weight in objective function; 0.1 by default")
		//-----------------------------------------
		("ff-activation-type", value<unsigned>()->default_value(1), "impose feed-forward activation type (1: GELU, 2: SWISH, 3: SWISH with learnable beta); 1 by default")
		//-----------------------------------------
		("position-encoding", value<unsigned>()->default_value(2), "impose positional encoding (0: none; 1: learned positional embedding; 2: sinusoid encoding); 2 by default")
		("max-pos-seq-len", value<unsigned>()->default_value(300), "n_ctx; specify the maximum word-based sentence length (either source or target) for learned positional encoding; 300 by default")
		//-----------------------------------------
		("use-hybrid-model", "use hybrid model in which RNN encodings are used in place of word embeddings and positional encodings (a hybrid architecture between AM and Transformer?) partially adopted from GNMT style; no by default")
		//-----------------------------------------		
		("attention-type", value<unsigned>()->default_value(1), "impose attention type (1: Luong attention type; 2: Bahdanau attention type); 1 by default")
		//-----------------------------------------
		("epochs,e", value<unsigned>()->default_value(20), "maximum number of training epochs")
		("patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved for early stopping; default none")
		//-----------------------------------------
		("lr-eta", value<float>()->default_value(0.1f), "SGD learning rate value (e.g., 0.1 for simple SGD trainer or smaller 0.001 for ADAM trainer)")
		("lr-eta-decay", value<float>()->default_value(2.0f), "SGD learning rate decay value")
		//-----------------------------------------
		("lr-epochs", value<unsigned>()->default_value(0), "no. of epochs for starting learning rate annealing (e.g., halving)") // learning rate scheduler 1
		("lr-patience", value<unsigned>()->default_value(0), "no. of times in which the model has not been improved, e.g., for starting learning rate annealing (e.g., halving)") // learning rate scheduler 2
		//-----------------------------------------
		("reset-if-stuck", "a strategy if the model gets stuck then reset everything and resume training; default not")
		("switch-to-adam", "switch to Adam trainer if getting stuck; default not")
		("use-smaller-minibatch", "use smaller mini-batch size if getting stuck; default not")
		("num-resets", value<unsigned>()->default_value(1), "no. of times the training process will be reset; default 1") 
		//-----------------------------------------
		("sampling", "sample during training; default not")
		//-----------------------------------------
		("average-checkpoints", value<unsigned>()->default_value(1), "specify number of checkpoints for model averaging; default single best model") // average checkpointing
		//-----------------------------------------
		("r2l-target", "use right-to-left direction for target during training; default not")
		//-----------------------------------------
		("treport", value<unsigned>()->default_value(50), "no. of training instances for reporting current model status on training data")
		("dreport", value<unsigned>()->default_value(5000), "no. of training instances for reporting current model status on development data (dreport = N * treport)")
		//-----------------------------------------
		("print-graphviz", "print graphviz-style computation graph; default not")
		//-----------------------------------------
		("verbose,v", "be extremely chatty")
		//-----------------------------------------
		("debug", "enable/disable simpler debugging by immediate computing mode or checking validity (refers to http://dynet.readthedocs.io/en/latest/debugging.html)")// for CPU only
		("dynet-profiling", value<int>()->default_value(0), "enable/disable auto profiling (https://github.com/clab/dynet/pull/1088/commits/bc34db98fa5e2e694f54f0e6b1d720d517c7530e)")// for debugging only			
	;
	
	store(parse_command_line(argc, argv, opts), vm); 
	if (vm.count("config") > 0)
	{
		ifstream config(vm["config"].as<std::string>().c_str());
		store(parse_config_file(config, opts), vm); 
	}
	notify(vm);

	// print command line
	cerr << endl << "PID=" << ::getpid() << endl;
	cerr << "Command: ";
	for (int i = 0; i < argc; i++){ 
		cerr << argv[i] << " "; 
	} 
	cerr << endl;
	
	// print help
	if (vm.count("help"))
	{
		cout << opts << "\n";
		return EXIT_FAILURE;
	}

	// hyper-parameters for training
	DEBUGGING_FLAG = vm.count("debug");
	VERBOSE = vm.count("verbose");
	TREPORT = vm["treport"].as<unsigned>(); 
	DREPORT = vm["dreport"].as<unsigned>(); 
	SAMPLING_TRAINING = vm.count("sampling");
	RESET_IF_STUCK = vm.count("reset-if-stuck");
	SWITCH_TO_ADAM = vm.count("switch-to-adam");
	USE_SMALLER_MINIBATCH = vm.count("use-smaller-minibatch");
	NUM_RESETS = vm["num-resets"].as<unsigned>();
	PRINT_GRAPHVIZ = vm.count("print-graphviz");
	MINIBATCH_SIZE = vm["minibatch-size"].as<unsigned>();
	bool is_training = !vm.count("test");

	// get and check model path
	std::string model_path = vm["model-path"].as<std::string>();
	struct stat sb;
	if (stat(model_path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))
		cerr << endl << "All model files will be loaded from: " << model_path << "." << endl;
	else
		TRANSFORMER_RUNTIME_ASSERT("The model-path does not exist!");

	// model recipe
	gpt_vocab d;
	SentinelMarkers sm;// sentinel markers
	WordIdSentences train_cor, devel_cor, test_cor;// integer-converted train and dev data
	transformer::TransformerConfig tfc;// Transformer's configuration (either loaded from file or newly-created)

	std::string config_file = model_path + "/model.config";// configuration file path

	// load fixed vocabularies from files
	std::string vocab_file = vm["vocab"].as<std::string>();
	assert(vocab_file.find("encoder.json") != string::npos);
	gpt_vocab_init(vocab_file, d);

	auto n_vocab = d.id_to_token.size();
	// TODO UNK
	sm._kTGT_SOS = n_vocab;
	sm._kTGT_EOS = n_vocab + 1;

	if (stat(config_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)){// check existence	
		// (incremental training)
		// to load the training profiles from previous training run
		cerr << "Found existing (trained) model from " << model_path << "!" << endl;	

		// load data files and config file for training
		if (is_training){
			if (!load_data_gpt2(vm, train_cor, devel_cor, test_cor, d))
				TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

			// model configuration
			ifstream inpf_cfg(config_file);
			assert(inpf_cfg);
		
			std::string line;
			getline(inpf_cfg, line);
			std::stringstream ss(line);
			tfc._tgt_vocab_size = d.size() + PAD_TOKENS;
			tfc._sm = sm;
			ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
			   >> tfc._decoder_emb_dropout_rate >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate 
			   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
			   >> tfc._position_encoding >> tfc._max_length
			   >> tfc._attention_type
			   >> tfc._ffl_activation_type
			   >> tfc._use_hybrid_model;
		}// otherwise for testing, move on!		
	}
	else{// not exist, meaning that the model will be created from scratch!
		cerr << "Preparing to train the model from scratch..." << endl;



		// load data files
		if (!load_data_gpt2(vm, train_cor, devel_cor, test_cor, d))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load data files!");

		// transformer configuration
		tfc = transformer::TransformerConfig(0, d.size()
			, vm["num-units"].as<unsigned>()
			, vm["num-heads"].as<unsigned>()
			, vm["nlayers"].as<unsigned>()
			, vm["n-ff-units-factor"].as<unsigned>()
			, 0.0f
			, 0.0f
			, vm["emb-dropout-p"].as<float>()
			, vm["sublayer-dropout-p"].as<float>()
			, vm["attention-dropout-p"].as<float>()
			, vm["ff-dropout-p"].as<float>()
			, vm.count("use-label-smoothing")
			, vm["label-smoothing-weight"].as<float>()
			, vm["position-encoding"].as<unsigned>()
			, 0
			, vm["max-pos-seq-len"].as<unsigned>()
			, sm
			, vm["attention-type"].as<unsigned>()
			, vm["ff-activation-type"].as<unsigned>()
			, false
			, vm.count("use-hybrid-model"));

		// save configuration file (for decoding/inference)
		std::string config_out_file = model_path + "/model.config";
		std::string params_out_file = model_path + "/model.params";
		save_config(config_out_file, params_out_file, tfc);		
	}

	if (is_training){
		// learning rate scheduler
		unsigned lr_epochs = vm["lr-epochs"].as<unsigned>(), lr_patience = vm["lr-patience"].as<unsigned>();
		if (lr_epochs > 0 && lr_patience > 0)
			cerr << "[WARNING] - Conflict on learning rate scheduler; use either lr-epochs or lr-patience!" << endl;		

		// initialise transformer object
		transformer::TransformerLModel tf(tfc, d);
		std::string model_file = model_path + "/model.params";
		if (stat(model_file.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
		{
			cerr << endl << "Loading pre-trained model from file: " << model_file << "..." << endl;
			tf.initialise_params_from_file(model_file);// load pre-trained model (for incremental training)
		}
		cerr << endl << "Count of model parameters: " << tf.get_model_parameters().parameter_count() << endl;

		// create SGD trainer
		Trainer* p_sgd_trainer = create_sgd_trainer(vm, tf.get_model_parameters());

		// train transformer-based language model
		run_train(tf
			, train_cor, devel_cor
			, p_sgd_trainer
			, model_path
			, vm["epochs"].as<unsigned>(), vm["patience"].as<unsigned>() /*early stopping*/
			, lr_epochs, vm["lr-eta-decay"].as<float>(), lr_patience/*learning rate scheduler*/
			, vm["average-checkpoints"].as<unsigned>());

		// clean up
		cerr << "Cleaning up..." << endl;
		delete p_sgd_trainer; 
		// transformer object will be automatically cleaned, no action required!
	}
	else{
		// load models
		std::string config_file = model_path + "/model.config";
		std::vector<std::shared_ptr<transformer::TransformerLModel>> v_tf_models;// to support ensemble models
		if (!load_model_config(config_file, v_tf_models, d, sm))
			TRANSFORMER_RUNTIME_ASSERT("Failed to load model(s)!");
		
		report_perplexity_score(v_tf_models, test_cor, MINIBATCH_SIZE);
	}

	return EXIT_SUCCESS;
}
//************************************************************************************************************************************************************

// ---
// load InstructionTuningDataset
bool load_data_gpt2(const variables_map& vm
	, WordIdSentences& train_cor, WordIdSentences& devel_cor, WordIdSentences& test_cor
	, const gpt_vocab& d)
{
	//! Actually contains whole dataset
	std::string train_path = vm["train"].as<std::vector<std::string>>()[0];
	assert(train_path.find("alpaca_gpt4_data.json") != string::npos);
	cerr << endl << "Reading training data from " << train_path << "...\n";

	// read file into string
	std::ifstream ifs(train_path);
	std::string json = std::string((std::istreambuf_iterator<char>(ifs)),
     		(std::istreambuf_iterator<char>()));
	// deser
	auto raw_data = parseDictsJson(json);
	auto ds_size = raw_data.size();
	// encode as List[sentence]
	vector<string> raw_sents;
	for (const auto &data_point: raw_data) {
		ostringstream oss;
		oss << "instruction " << data_point.at("instruction") << "\n\ninput " << data_point.at("input") << "\n\noutput " << data_point.at("output") << "\n\n";
		raw_sents.push_back(oss.str());
	}

	// train/test split
    size_t train_size = ds_size * 4 / 5;
    size_t val_size = ds_size / 10;
    size_t test_size = ds_size - train_size - val_size;
	auto val_begin = train_size;
	auto val_end = val_begin + val_size;

	// limit the percent of training data to be used
	unsigned train_percent = vm["train-percent"].as<unsigned>();
	if (train_percent < 100 && train_percent > 0) {
		cerr << "Only use " << train_percent << "% of " << train_size << " training instances: ";
		train_size = train_size * train_percent / 100;
		cerr << train_size << " instances remaining!" << endl;
	} else if (train_percent != 100){
		cerr << "Invalid --train-percent <num> used. <num> must be (0,100]" << endl;
		return false;
	}

	// Tokenize. Deferred here because it's slow
	int lc = 0, toks = 0;
	unsigned int max_len = 0;
	unsigned max_slen = vm["max-seq-len"].as<unsigned>();

	for (int i=0, e=raw_sents.size(); i<e; ++i) {
		if (i>=train_size && i<val_begin) continue;
		const auto& raw_sent = raw_sents[i];
		WordIdSentence sent = gpt_tokenize(d, raw_sent);
		if (max_slen > 0 && sent.size() > max_slen) continue;
		// int sos_id = d.special_tokens[0], eos_id = d.special_tokens[1];
		if (sent.size() < 1) continue;

		if (i<train_size) train_cor.push_back(sent);
		else if (i<val_end) devel_cor.push_back(sent);
		else test_cor.push_back(sent);

		max_len = std::max(max_len, (unsigned int)sent.size());
		toks += sent.size();
		++lc;
	}
	cerr << lc << " lines, " << toks << " tokens, " << "max length: " << max_len << ", " << d.size() << " types" << endl;

	if (DREPORT >= train_cor.size())
		cerr << "WARNING: --dreport <num> (" << DREPORT << ")" << " is too large, <= training data size (" << train_cor.size() << ")" << endl;

	// TODO set up <unk> ids

	return true;
}
// ---

// ---
dynet::Trainer* create_sgd_trainer(const variables_map& vm, dynet::ParameterCollection& model){
	// setup SGD trainer
	Trainer* sgd = nullptr;
	unsigned sgd_type = vm["sgd-trainer"].as<unsigned>();
	if (sgd_type == 1)
		sgd = new MomentumSGDTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 2)
		sgd = new AdagradTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 3)
		sgd = new AdadeltaTrainer(model);
	else if (sgd_type == 4)
		sgd = new AdamTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 5)
		sgd = new RMSPropTrainer(model, vm["lr-eta"].as<float>());
	else if (sgd_type == 0)//Vanilla SGD trainer
		sgd = new SimpleSGDTrainer(model, vm["lr-eta"].as<float>());
	else
	   	TRANSFORMER_RUNTIME_ASSERT("Unknown SGD trainer type! (0: vanilla SGD; 1: momentum SGD; 2: Adagrad; 3: AdaDelta; 4: Adam; 5: RMSProp)");
	sgd->clip_threshold = vm["grad-clip-threshold"].as<float>();// * MINIBATCH_SIZE;// use larger gradient clipping threshold if training with mini-batching, correct?
	sgd->sparse_updates_enabled = vm["sparse-updates"].as<bool>();
	if (!sgd->sparse_updates_enabled)
		cerr << "Sparse updates for lookup parameter(s) to be disabled!" << endl;

	return sgd;
}
// ---

// ---
bool load_model_config(const std::string& model_cfg_file
	, std::vector<std::shared_ptr<transformer::TransformerLModel>>& v_models
	, gpt_vocab& d
	, const transformer::SentinelMarkers& sm)
{
	cerr << "Loading model(s) from configuration file: " << model_cfg_file << "..." << endl;

	v_models.clear();

	ifstream inpf(model_cfg_file);
	assert(inpf);
	
	unsigned i = 0;
	std::string line;
	while (getline(inpf, line)){
		if ("" == line) break;

		// each line has the format: 
		// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <emb-dropout> <sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <your-trained-model-path>
		// e.g.,
		// 128 2 2 4 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.lm.transformer.h2_l2_u128_do01010101_att1_ls01_pe1_ml300_ffrelu_run1
		// 128 2 2 4 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.lm.transformer.h2_l2_u128_do01010101_att1_ls01_pe1_ml300_ffrelu_run2
		cerr << "Loading model " << i+1 << "..." << endl;
		std::stringstream ss(line);

		transformer::TransformerConfig tfc;
		std::string model_file;

		tfc._tgt_vocab_size = d.size() + PAD_TOKENS;
		tfc._sm = sm;
		
		ss >> tfc._num_units >> tfc._nheads >> tfc._nlayers >> tfc._n_ff_units_factor
		   >> tfc._decoder_emb_dropout_rate >> tfc._decoder_sublayer_dropout_rate >> tfc._attention_dropout_rate >> tfc._ff_dropout_rate 
		   >> tfc._use_label_smoothing >> tfc._label_smoothing_weight
		   >> tfc._position_encoding >> tfc._max_length
		   >> tfc._attention_type
		   >> tfc._ffl_activation_type
		   >> tfc._use_hybrid_model;		
		ss >> model_file;
		tfc._is_training = false;
		tfc._use_dropout = false;

		v_models.push_back(std::shared_ptr<transformer::TransformerLModel>());
		v_models[i].reset(new transformer::TransformerLModel(tfc, d));
		v_models[i].get()->initialise_params_from_file(model_file);// load pre-trained model from file
		cerr << "Count of model parameters: " << v_models[i].get()->get_model_parameters().parameter_count() << endl;

		i++;
	}

	cerr << "Done!" << endl << endl;

	return true;
}
// ---

// ---
void report_perplexity_score(std::vector<std::shared_ptr<transformer::TransformerLModel>>& v_tf_models, WordIdSentences &test_cor, unsigned minibatch_size)// support ensemble models
{
	// Sentinel symbols
	const transformer::SentinelMarkers& sm = v_tf_models[0].get()->get_config()._sm;

	transformer::ModelStats dstats;
	get_dev_stats(test_cor, sm._kTGT_UNK, dstats);

	std::vector<WordIdSentences> test_cor_minibatch;
	cerr << "Creating minibatches for testing data (using minibatch_size=" << minibatch_size << ")..." << endl;
	create_minibatches(test_cor, minibatch_size, test_cor_minibatch);// for dev

	MyTimer timer_iteration("completed in");

	unsigned cb = 0;
	float dloss = 0.f;
	for (auto& minibatch : test_cor_minibatch) {
		cerr << "Processing minibatch " << cb++ << "..." << endl;

		unsigned bsize = minibatch.size();
		size_t max_len = minibatch[0].size();
		for(size_t i = 1; i < bsize; i++) max_len = std::max(max_len, minibatch[i].size());
		
		dynet::ComputationGraph cg;
		WordIdSentences partial_sents(bsize, WordIdSentence(1, sm._kTGT_SOS));
		std::vector<unsigned> next_words(bsize);
		for (unsigned t = 1; t < max_len; t++){// shifted to the right
			for (unsigned bs = 0; bs < bsize; bs++)
				next_words[bs] = (minibatch[bs].size()>t)?(unsigned)minibatch[bs][t]:sm._kTGT_EOS;

			// perform the forward step on all models
			std::vector<Expression> i_softmaxes, i_aligns/*unused*/;
			for(int j : boost::irange(0, (int)v_tf_models.size())){
				i_softmaxes.push_back(v_tf_models[j].get()->step_forward(cg
					, partial_sents
					, false
					, i_aligns));
			}

			// ToDo: losses for padded positions should not be included. This will result in smaller perplexity scores!
			dynet::Expression i_logprob = dynet::log({dynet::average(i_softmaxes)});// averaging
			dynet::Expression i_loss = dynet::sum_batches(dynet::pick(-i_logprob, next_words));
			dloss += dynet::as_scalar(cg.incremental_forward(i_loss));

			for (unsigned bs = 0; bs < bsize; bs++)
				partial_sents[bs].push_back(next_words[bs]);

			cg.clear();
		}
	}

	dstats._scores[1] = dloss;

	cerr << "--------------------------------------------------------------------------------------------------------" << endl;
	cerr << "***TEST: " << "sents=" << test_cor.size() << " words=" << dstats._words_tgt << " unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string(true) << ' ';

	timer_iteration.show();
}
// ---

// ---
void get_dev_stats(const WordIdSentences &devel_cor
	, int unk_id
	, transformer::ModelStats& dstats) // ToDo: support batch?
{
	for (unsigned i = 0; i < devel_cor.size(); ++i) {
		WordIdSentence dsent = devel_cor[i];  
		dstats._words_tgt += dsent.size() - 1; // shifted right 
		for (auto& word : dsent) if (word == unk_id) dstats._words_tgt_unk++;
	}
}
// ---

// ---
void run_train(transformer::TransformerLModel &tf, WordIdSentences &train_cor, WordIdSentences &devel_cor, 
	dynet::Trainer* &p_sgd, 
	const std::string& model_path,
	unsigned max_epochs, unsigned patience, 
	unsigned lr_epochs, float lr_eta_decay, unsigned lr_patience,
	unsigned average_checkpoints)
{
	std::string params_out_file = model_path + "/model.params";

	// model configuration
	const transformer::TransformerConfig& tfc = tf.get_config();

	// get current dict
	auto& dict = tf.get_dict();

	// create minibatches
	std::vector<WordIdSentences> train_cor_minibatch, dev_cor_minibatch;
	std::vector<size_t> train_ids_minibatch;
	size_t minibatch_size = MINIBATCH_SIZE;
	cerr << endl << "Creating minibatches for training data (using minibatch_size=" << minibatch_size << ")..." << endl;
	create_minibatches(train_cor, minibatch_size, train_cor_minibatch);// for train
	cerr << "Creating minibatches for development data (using minibatch_size=" << minibatch_size << ")..." << endl;
	create_minibatches(devel_cor, minibatch_size, dev_cor_minibatch);// for dev
	// create a sentence list for this minibatch
	train_ids_minibatch.resize(train_cor_minibatch.size());
	std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

	// stats on dev  
	transformer::ModelStats dstats;
	get_dev_stats(devel_cor, tfc._sm._kTGT_UNK, dstats);
	
	unsigned report_every_i = TREPORT;
	unsigned dev_every_i_reports = DREPORT;

	// shuffle minibatches
	cerr << endl << "***SHUFFLE" << endl;
	std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

	unsigned sid = 0, id = 0, last_print = 0;
	MyTimer timer_epoch("completed in"), timer_iteration("completed in");
	unsigned epoch = 0, cpt = 0/*count of patience*/;
	while (epoch < max_epochs) {
		transformer::ModelStats tstats;

		tf.set_dropout(true);// enable dropout

		for (unsigned iter = 0; iter < dev_every_i_reports;) {
			if (id == train_ids_minibatch.size()) { 
				//timing
				cerr << "***Epoch " << epoch << " is finished. ";
				timer_epoch.show();

				epoch++;

				id = 0;
				sid = 0;
				last_print = 0;

				// learning rate scheduler 1: after lr_epochs, for every next epoch, the learning rate will be decreased by a factor of eta_decay.
				if (lr_epochs > 0 && epoch >= lr_epochs)
					p_sgd->learning_rate /= lr_eta_decay; 

				if (epoch >= max_epochs) break;

				// shuffle the access order
				cerr << "***SHUFFLE" << endl;
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);				

				timer_epoch.reset();
			}

			// build graph for this instance
			auto pCg = std::make_shared<dynet::ComputationGraph>();
			dynet::ComputationGraph& cg = *pCg;// dynamic computation graph for each data batch
			if (DEBUGGING_FLAG){//http://dynet.readthedocs.io/en/latest/debugging.html
				cg.set_immediate_compute(true);
				cg.set_check_validity(true);
			}
	
			transformer::ModelStats ctstats;
			Expression i_xent = tf.build_graph(cg, train_cor_minibatch[train_ids_minibatch[id]], &ctstats);
	
			if (PRINT_GRAPHVIZ) {
				cerr << "***********************************************************************************" << endl;
				cg.print_graphviz();
				cerr << "***********************************************************************************" << endl;
			}

			Expression i_objective = i_xent;

			// perform forward computation for aggregate objective
			cg.forward(i_objective);

			// grab the parts of the objective
			float loss = as_scalar(cg.get_value(i_xent.i));
			if (!is_valid(loss)){
				std::cerr << "***Warning***: nan or -nan values occurred!" << std::endl;
				++id;
				continue;
			}

			tstats._scores[1] += loss;
			tstats._words_tgt += ctstats._words_tgt;
			tstats._words_tgt_unk += ctstats._words_tgt_unk;  

			cg.backward(i_objective);
			p_sgd->update();

			sid += train_cor_minibatch[train_ids_minibatch[id]].size();
			iter += train_cor_minibatch[train_ids_minibatch[id]].size();

			if (sid / report_every_i != last_print 
					|| iter >= dev_every_i_reports
					|| id + 1 == train_ids_minibatch.size()){
				last_print = sid / report_every_i;

				float elapsed = timer_iteration.elapsed();

				p_sgd->status();
				cerr << "sents=" << sid << " ";
				cerr /*<< "loss=" << tstats._scores[1]*/ << "words=" << tstats._words_tgt << " unks=" << tstats._words_tgt_unk << " " << tstats.get_score_string() << ' ';//<< " E=" << (tstats._scores[1] / tstats._words_tgt) << " ppl=" << exp(tstats._scores[1] / tstats._words_tgt) << ' ';
				cerr /*<< "time_elapsed=" << elapsed*/ << "(" << (float)(tstats._words_tgt) * 1000.f / elapsed << " words/sec)" << endl;  					
			}
			   		 
			++id;
		}
		// goto finish; // TODO remove before production
		// show score on dev data?
		tf.set_dropout(false);// disable dropout for evaluating dev data

		// sample a random sentence (for observing translations during training progress)
		assert((SAMPLING_TRAINING == false) && "Not implemented");

		timer_iteration.reset();

		// compute cross entropy loss (xent)
		/* non-batched version
		for (unsigned i = 0; i < devel_cor.size(); ++i) {
			WordIdSentence dsent = devel_cor[i];  

			dynet::ComputationGraph cg;
			auto i_xent = tf.build_graph(cg, WordIdSentences(1, dsent), nullptr, true);
			dstats._scores[1] += as_scalar(cg.forward(i_xent));
		}*/
		// batched version (faster)
		float dloss = 0.f;
		for (const WordIdSentences& dsentb : dev_cor_minibatch){
			dynet::ComputationGraph cg;
			auto i_xent = tf.build_graph(cg, dsentb, nullptr, true);
			dloss += as_scalar(cg.incremental_forward(i_xent));
		}
		
		float elapsed = timer_iteration.elapsed();

		dstats._scores[1] = dloss;
		dstats.update_best_score(cpt);
		if (cpt == 0){
			// FIXME: consider average checkpointing?
			tf.save_params_to_file(params_out_file);
		}
		
		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
		cerr << "***DEV [epoch=" << (float)epoch + (float)sid/(float)train_cor.size() << " eta=" << p_sgd->learning_rate << "]" << " sents=" << devel_cor.size( )<< " words=" << dstats._words_tgt << " unks=" << dstats._words_tgt_unk << " " << dstats.get_score_string() << ' ';
		if (cpt > 0) cerr << "(not improved, best ppl on dev so far: " << dstats.get_score_string(false)  << ") ";
		cerr << "[completed in " << elapsed << " ms]" << endl;
	
		// learning rate scheduler 2: if the model has not been improved for lr_patience times, decrease the learning rate by lr_eta_decay factor.
		if (lr_patience > 0 && cpt > 0 && cpt % lr_patience == 0){
			cerr << "The model has not been improved for " << lr_patience << " times. Decreasing the learning rate..." << endl;
			p_sgd->learning_rate /= lr_eta_decay;
		}

		// another early stopping criterion
		if (patience > 0 && cpt >= patience)
		{
			if (RESET_IF_STUCK){
				cerr << "The model seems to get stuck. Resetting now...!" << endl;
				cerr << "Attempting to resume the training..." << endl;			
				// 1) load the previous best model
				cerr << "Loading previous best model..." << endl;
				tf.initialise_params_from_file(params_out_file);
				// 2) some useful tricks:
				sid = 0; id = 0; last_print = 0; cpt = 0;
				// a) reset SGD trainer, switching to Adam instead!
				if (SWITCH_TO_ADAM){ 
					delete p_sgd; p_sgd = 0;
					p_sgd = new dynet::AdamTrainer(tf.get_model_parameters(), 0.001f/*maybe smaller?*/);
					SWITCH_TO_ADAM = false;// do it once!
				}
				// b) use smaller batch size
				if (USE_SMALLER_MINIBATCH){
					cerr << "Creating minibatches for training data (using minibatch_size=" << minibatch_size/2 << ")..." << endl;
					train_cor_minibatch.clear();
					train_ids_minibatch.clear();
					create_minibatches(train_cor, minibatch_size/2, train_cor_minibatch);// for train
					// create a sentence list for this train minibatch
					train_ids_minibatch.resize(train_cor_minibatch.size());
					std::iota(train_ids_minibatch.begin(), train_ids_minibatch.end(), 0);

					minibatch_size /= 2;
					report_every_i /= 2;
				}
				// 3) shuffle the training data
				cerr << "***SHUFFLE" << endl;
				std::shuffle(train_ids_minibatch.begin(), train_ids_minibatch.end(), *dynet::rndeng);

				NUM_RESETS--;
				if (NUM_RESETS == 0)
					RESET_IF_STUCK = false;// it's right time to stop anyway!
			}			
			else{
				cerr << "The model has not been improved for " << patience << " times. Stopping now...!" << endl;
				cerr << "No. of epochs so far: " << epoch << "." << endl;
				cerr << "Best ppl on dev: " << dstats.get_score_string(false) << endl;
				cerr << "--------------------------------------------------------------------------------------------------------" << endl;

				break;
			}
		}

		timer_iteration.reset();

		cerr << "--------------------------------------------------------------------------------------------------------" << endl;
	}
finish:
	cerr << endl << "Transformer training completed!" << endl;
	DAO::profiler.dump(model_path + "/train");
}
// ---

//---
void save_config(const std::string& config_out_file, const std::string& params_out_file, const TransformerConfig& tfc)
{
	// each line has the format: 
	// <num-units> <num-heads> <nlayers> <ff-num-units-factor> <emb-dropout> <sublayer-dropout> <attention-dropout> <ff-dropout> <use-label-smoothing> <label-smoothing-weight> <position-encoding-type> <max-seq-len> <attention-type> <ff-activation-type> <use-hybrid-model> <your-trained-model-path>
	// e.g.,
	// 128 2 2 4 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.lm.transformer.h2_l2_u128_do01010101_att1_ls01_pe1_ml300_ffrelu_run1
	// 128 2 2 4 0.1 0.1 0.1 0.1 0 0.1 1 0 300 1 1 0 0 <your-path>/models/iwslt-envi/params.lm.transformer.h2_l2_u128_do01010101_att1_ls01_pe1_ml300_ffrelu_run2
	std::stringstream ss;
		
	ss << tfc._num_units << " " << tfc._nheads << " " << tfc._nlayers << " " << tfc._n_ff_units_factor << " "
		<< tfc._decoder_emb_dropout_rate << " " << tfc._decoder_sublayer_dropout_rate << " " << tfc._attention_dropout_rate << " " << tfc._ff_dropout_rate << " "
		<< tfc._use_label_smoothing << " " << tfc._label_smoothing_weight << " "
		<< tfc._position_encoding << " " << tfc._max_length << " "
		<< tfc._attention_type << " "
		<< tfc._ffl_activation_type << " "
		<< tfc._use_hybrid_model << " ";		
	ss << params_out_file;

	ofstream outf_cfg(config_out_file);
	assert(outf_cfg);
	outf_cfg << ss.str();
}
//---

//---
std::string get_sentence(const WordIdSentence& source, Dict& d){
	std::stringstream ss;
	for (WordId w : source){
		ss << d.convert(w) << " ";
	}

	return ss.str();
}
//---

