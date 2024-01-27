#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <regex>

#include "dynet/dict.h"

using namespace std;
using namespace dynet;

WordIdCorpus read_corpus(const string &filename
	, dynet::Dict* sd, dynet::Dict* td
	, bool cid=true/*corpus id, 1:train;0:otherwise*/
	, unsigned slen=0, bool r2l_target=false
	, bool swap=false);

WordIdCorpusWithMeta read_corpus(const string &filename, const string &side_filename
	, dynet::Dict* sd, dynet::Dict* td, dynet::Dict* side_d
	, bool cid=true/*corpus id, 1:train;0:otherwise*/
	, unsigned slen=0, bool r2l_target=false
	, bool swap=false);

WordIdSentences read_corpus(const string &filename
	, dynet::Dict* d
	, bool cid=true/*corpus id, 1:train;0:otherwise*/
	, unsigned slen=0, bool r2l_target=false);

WordIdCorpus read_corpus(const string &filename
	, dynet::Dict* sd, dynet::Dict* td
	, bool cid
	, unsigned slen, bool r2l_target
	, bool swap)
{
	bool use_joint_vocab = false;
	if (sd == td) use_joint_vocab = true;

	int kSRC_SOS = sd->convert("<s>");
	int kSRC_EOS = sd->convert("</s>");
	int kTGT_SOS = td->convert("<s>");
	int kTGT_EOS = td->convert("</s>");

	ifstream in(filename);
	assert(in);

	WordIdCorpus corpus;

	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	unsigned int max_src_len = 0, max_tgt_len = 0;
	while (getline(in, line)) {
		WordIdSentence source, target;

		if (!swap)
			read_sentence_pair(line, source, *sd, target, *td);
		else read_sentence_pair(line, source, *td, target, *sd);

		// reverse the target if required
		if (r2l_target) 
			std::reverse(target.begin() + 1/*BOS*/, target.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (source.size() > slen || target.size() > slen)
				continue;// ignore this sentence
		}

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			stringstream ss;
			ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			assert(ss.str().c_str());

			abort();
		}

		if (source.size() < 3 || target.size() < 3){ // ignore empty sentences, e.g., <s> </s>
			continue;
		}

		corpus.push_back(WordIdSentencePair(source, target));

		max_src_len = std::max(max_src_len, (unsigned int)source.size());
		max_tgt_len = std::max(max_tgt_len, (unsigned int)target.size());

		stoks += source.size();
		ttoks += target.size();

		++lc;
	}

	// print stats
	if (cid){
		if (!use_joint_vocab)
			cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " & " << td->size() << " types" << endl;
		else 
			cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " joint s & t types" << endl;
	}
	else 
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << endl;

	return corpus;
}

WordIdCorpusWithMeta read_corpus(const string &filename, const string &side_filename
	, dynet::Dict* sd, dynet::Dict* td, dynet::Dict* side_d
	, bool cid
	, unsigned slen, bool r2l_target
	, bool swap)
{
	bool use_joint_vocab = false;
	if (sd == td) use_joint_vocab = true;

	int kSRC_SOS = sd->convert("<s>");
	int kSRC_EOS = sd->convert("</s>");
	int kTGT_SOS = td->convert("<s>");
	int kTGT_EOS = td->convert("</s>");

	ifstream in(filename);
	assert(in);

	ifstream in_side(side_filename);
	assert(in_side);

	WordIdCorpusWithMeta corpus;

	string line, line_side;
	int lc = 0, stoks = 0, ttoks = 0;
	unsigned int max_src_len = 0, max_tgt_len = 0;
	while (getline(in, line) && getline(in_side, line_side)) {
		WordIdSentence source, target, side;

		if (!swap)
			read_sentence_pair(line, source, *sd, target, *td);
		else read_sentence_pair(line, source, *td, target, *sd);

		side = read_sentence(line_side, *side_d);

		// reverse the target if required
		if (r2l_target) 
			std::reverse(target.begin() + 1/*BOS*/, target.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (source.size() > slen || target.size() > slen)
				continue;// ignore this sentence
		}

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			stringstream ss;
			ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			assert(ss.str().c_str());

			abort();
		}

		if (source.size() < 3 || target.size() < 3){ // ignore empty sentences, e.g., <s> </s>
			continue;
		}

		corpus.push_back(WordIdSentenceTriple(source, target, side));

		max_src_len = std::max(max_src_len, (unsigned int)source.size());
		max_tgt_len = std::max(max_tgt_len, (unsigned int)target.size());

		stoks += source.size();
		ttoks += target.size();

		++lc;
	}

	// print stats
	if (cid){
		if (!use_joint_vocab)
			cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " & " << td->size() << " types" << endl;
		else 
			cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " joint s & t types" << endl;
	}
	else 
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << endl;

	return corpus;
}

WordIdSentences read_corpus(const string &filename
	, dynet::Dict* d
	, bool cid
	, unsigned slen, bool r2l)
{
	int SOS = d->convert("<s>");
	int EOS = d->convert("</s>");

	ifstream in(filename);
	assert(in);

	WordIdSentences corpus;

	string line;
	int lc = 0, toks = 0;
	unsigned int max_len = 0;
	while (getline(in, line)) {
		WordIdSentence sent = read_sentence(line, *d);

		// reverse the target if required
		if (r2l) 
			std::reverse(sent.begin() + 1/*BOS*/, sent.end() - 1/*EOS*/);

		// constrain sentence length(s)
		if (cid/*train only*/ && slen > 0/*length limit*/){
			if (sent.size() > slen)
				continue;// ignore this sentence
		}

		if (sent.front() != SOS && sent.back() != EOS) {
			stringstream ss;
			ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
			assert(ss.str().c_str());

			abort();
		}

		if (sent.size() < 3){ // ignore empty sentences, e.g., <s> </s>
			continue;
		}

		corpus.push_back(sent);

		max_len = std::max(max_len, (unsigned int)sent.size());

		toks += sent.size();

		++lc;
	}

	// print stats
	if (cid)
		cerr << lc << " lines, " << toks << " tokens, " << "max length: " << max_len << ", " << d->size() << " types" << endl;
	else 
		cerr << lc << " lines, " << toks << " tokens, " << "max length: " << max_len << endl;

	return corpus;
}

// Specific parser for alpaca_gpt4_data.json, which is an array of dictionaries
std::vector<std::map<std::string, std::string>> parseDictsJson(const std::string& json) {
    std::vector<std::map<std::string, std::string>> result;
	result.reserve(53000); // len(dataset) == 52002
	// std::regex unicodePattern("\\\\u[0-9a-fA-F]{4}");
	// std::regex escapePattern(R"(\\(.))");

    size_t pos = 0;
    while ((pos = json.find('{', pos)) != std::string::npos) {
        size_t endPos = json.find('\n', pos);
		// We know each dict has three entries
		endPos = json.find('\n', endPos + 1);
		endPos = json.find('\n', endPos + 1);
		endPos = json.find('\n', endPos + 1);
		// find end of dict
		endPos = json.find('}', endPos + 1);
        assert((endPos != std::string::npos) && "Malformed JSON");

        std::string dictionary = json.substr(pos + 1, endPos - pos - 1);
        std::istringstream iss(dictionary);
        std::string key, value;
        std::map<std::string, std::string> dict;

        while (std::getline(iss, key, ':') && std::getline(iss, value, '\n')) {
            // Remove leading/trailing whitespaces and quotes
            key.erase(0, key.find('\"') + 1);
            key.erase(key.rfind('\"'));

            value.erase(0, value.find('\"') + 1);
            value.erase(value.rfind('\"'));

			// XXX slow + not all prompts are escaped
			// value = std::regex_replace(value, unicodePattern, " ");
			// value = std::regex_replace(value, escapePattern, "$1");

            dict[key] = value;
        }

        result.push_back(dict);
		if (result.size() % 10000 == 0) std::cerr << "parsed " << result.size() << " records.\n";
        pos = endPos + 1;
    }

    return result;
}