// bin/decode-faster-mapped.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder-coupled.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Decode, reading log-likelihoods as matrices\n"
        " (model is needed only for the integer mappings in its transition-model)\n"
        "Usage:   decode-faster-mapped [options] <model-1-in> <model-2-in> <fst-1-in> <fst-2-in> "
        "<loglikes-1-rspecifier> <loglikes-2-rspecifier> <words-1-wspecifier> <words-2-wspecifier> [<alignments-1-wspecifier> <alignments-2-wspecifier>]\n";
    ParseOptions po(usage);
    bool binary = true;
    BaseFloat acoustic_scale_1 = 0.1;
    BaseFloat acoustic_scale_2 = 0.1;
    bool allow_partial = true;
    std::string word_syms_filename_1;
    std::string word_syms_filename_2;
    std::string coupled_lm_filename_1;
    std::string coupled_lm_filename_2;
    FasterDecoderCoupledOptions decoder_opts;
    decoder_opts.Register(&po, true);  // true == include obscure settings.
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("acoustic-scale-1", &acoustic_scale_1, "Scaling factor for acoustic likelihoods 1");
    po.Register("acoustic-scale-2", &acoustic_scale_2, "Scaling factor for acoustic likelihoods 2");
    po.Register("allow-partial", &allow_partial, "Produce output even when final state was not reached");
    po.Register("word-symbol-table-1", &word_syms_filename_1, "Symbol table for words 1 [for debug output]");
    po.Register("word-symbol-table-2", &word_syms_filename_2, "Symbol table for words 2 [for debug output]");

    po.Read(argc, argv);

    if (po.NumArgs() < 7 || po.NumArgs() > 8) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename_1 = po.GetArg(1),
	model_in_filename_2 = po.GetArg(2),
        fst_in_filename_1 = po.GetArg(3),
	fst_in_filename_2 = po.GetArg(4),
        loglikes_rspecifier_1 = po.GetArg(5),
	loglikes_rspecifier_2 = po.GetArg(6),
        words_wspecifier_1 = po.GetArg(7),
	words_wspecifier_2 = po.GetArg(8),
        alignment_wspecifier_1 = po.GetOptArg(9),
    	alignment_wspecifier_2 = po.GetOptArg(10);

    TransitionModel trans_model_1;
    ReadKaldiObject(model_in_filename_1, &trans_model_1);
    TransitionModel trans_model_2;
    ReadKaldiObject(model_in_filename_2, &trans_model_2);

    Int32VectorWriter words_writer_1(words_wspecifier_1);
    Int32VectorWriter words_writer_2(words_wspecifier_2);

    Int32VectorWriter alignment_writer_1(alignment_wspecifier_1);
    Int32VectorWriter alignment_writer_2(alignment_wspecifier_2);

    fst::SymbolTable *word_syms_1 = NULL;
    if (word_syms_filename_1 != "") {
      word_syms_1 = fst::SymbolTable::ReadText(word_syms_filename_1);
      if (!word_syms_1)
        KALDI_ERR << "Could not read symbol table from file "<<word_syms_filename_1;
    }

    fst::SymbolTable *word_syms_2 = NULL;
    if (word_syms_filename_2 != "") {
      word_syms_2 = fst::SymbolTable::ReadText(word_syms_filename_2);
      if (!word_syms_2)
        KALDI_ERR << "Could not read symbol table from file "<<word_syms_filename_2;
    }

    SequentialBaseFloatMatrixReader loglikes_reader_1(loglikes_rspecifier_1);
    SequentialBaseFloatMatrixReader loglikes_reader_2(loglikes_rspecifier_2);

    // It's important that we initialize decode_fst after loglikes_reader, as it
    // can prevent crashes on systems installed without enough virtual memory.
    // It has to do with what happens on UNIX systems if you call fork() on a
    // large process: the page-table entries are duplicated, which requires a
    // lot of virtual memory.
    Fst<StdArc> *decode_fst_1 = fst::ReadFstKaldiGeneric(fst_in_filename_1);
    Fst<StdArc> *decode_fst_2 = fst::ReadFstKaldiGeneric(fst_in_filename_2);

    BaseFloat tot_like_1 = 0.0;
    BaseFloat tot_like_2 = 0.0;
    // Coupled images have the same number of frames
    kaldi::int64 frame_count_1 = 0;
    kaldi::int64 frame_count_2 = 0;
    int num_success = 0, num_fail = 0;
    FasterDecoderCoupled decoder(*decode_fst_1, *decode_fst_2, decoder_opts);

    Timer timer;

    for (; !loglikes_reader_1.Done() && !loglikes_reader_2.Done(); loglikes_reader_1.Next(), loglikes_reader_2.Next()) {
      std::string key_1 = loglikes_reader_1.Key();
      std::string key_2 = loglikes_reader_2.Key();
      const Matrix<BaseFloat> &loglikes_1 (loglikes_reader_1.Value());
      const Matrix<BaseFloat> &loglikes_2 (loglikes_reader_2.Value());

      if (loglikes_1.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key_1;
        num_fail++;
        continue;
      }

      if (loglikes_2.NumRows() == 0) {
        KALDI_WARN << "Zero-length utterance: " << key_2;
	num_fail++;
	continue;
      }

      DecodableMatrixScaledMapped decodable_1(trans_model_1, loglikes_1, acoustic_scale_1);
      DecodableMatrixScaledMapped decodable_2(trans_model_2, loglikes_2, acoustic_scale_2);
      decoder.Decode(&decodable_1, &decodable_2);

      VectorFst<LatticeArc> decoded_1;  // linear FST.
      VectorFst<LatticeArc> decoded_2;  // linear FST.

      // ARREGLAR EL IF DE REACHED FINAL
      if ( (allow_partial || (decoder.ReachedFinal_1() && decoder.ReachedFinal_2()))
           && decoder.GetBestPath_1(&decoded_1) && decoder.GetBestPath_2(&decoded_2) ) {
        num_success++;
        if (!decoder.ReachedFinal_1() || !decoder.ReachedFinal_2())
          KALDI_WARN << "Decoder did not reach end-state, outputting partial traceback.";

        std::vector<int32> alignment_1;
	std::vector<int32> alignment_2;
        std::vector<int32> words_1;
	std::vector<int32> words_2;
        LatticeWeight weight_1;
	LatticeWeight weight_2;
        frame_count_1 += loglikes_1.NumRows();
	frame_count_2 += loglikes_2.NumRows();

        GetLinearSymbolSequence(decoded_1, &alignment_1, &words_1, &weight_1);
	GetLinearSymbolSequence(decoded_2, &alignment_2, &words_2, &weight_2);

        words_writer_1.Write(key_1, words_1);
	words_writer_2.Write(key_2, words_2);

        if (alignment_writer_1.IsOpen())
          alignment_writer_1.Write(key_1, alignment_1);
	if (alignment_writer_2.IsOpen())
          alignment_writer_2.Write(key_2, alignment_2);

        if (word_syms_1 != NULL) {
          std::cerr << key_1 << ' ';
          for (size_t i = 0; i < words_1.size(); i++) {
            std::string s = word_syms_1->Find(words_1[i]);
            if (s == "")
              KALDI_ERR << "Word-id " << words_1[i] <<" not in symbol table.";
            std::cerr << s << ' ';
          }
          std::cerr << '\n';
        }

	if (word_syms_2 != NULL) {
	  std::cerr << key_2 << ' ';
	  for (size_t i = 0; i < words_2.size(); i++) {
	    std::string s = word_syms_2->Find(words_1[i]);
	    if (s == "")
	      KALDI_ERR << "Word-id " << words_2[i] <<" not in symbol table.";
	    std::cerr << s << ' ';
	  }
	  std::cerr << '\n';
	}

        BaseFloat like_1 = -weight_1.Value1() -weight_1.Value2();
	BaseFloat like_2 = -weight_2.Value1() -weight_2.Value2();
        tot_like_1 += like_1;
	tot_like_2 += like_2;
        KALDI_LOG << "Log-like per frame for utterance " << key_1 << " is "
                  << (like_1 / loglikes_1.NumRows()) << " over "
                  << loglikes_1.NumRows() << " frames.";
	KALDI_LOG << "Log-like per frame for utterance " << key_2 << " is "
		  << (like_2 / loglikes_2.NumRows()) << " over "
		  << loglikes_2.NumRows() << " frames.";

      } else {
        num_fail++;
        KALDI_WARN << "Did not successfully decode utterance " << key_1
                   << ", len = " << loglikes_1.NumRows();
	KALDI_WARN << "Did not successfully decode utterance " << key_2
		   << ", len = " << loglikes_2.NumRows();
      }
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count_1);
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
	      << "s: real-time factor assuming 100 frames/sec is "
	      << (elapsed*100.0/frame_count_2);
    KALDI_LOG << "Done " << num_success << " pairs of utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like_1/frame_count_1)
              << " over " << frame_count_1 << " frames.";
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like_2/frame_count_2)
	      << " over " << frame_count_2 << " frames.";

    delete word_syms_1;
    delete word_syms_2;
    delete decode_fst_1;
    delete decode_fst_2;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


