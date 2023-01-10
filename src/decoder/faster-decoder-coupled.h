// decoder/faster-decoder.h

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

#ifndef KALDI_DECODER_FASTER_DECODER_COUPLED_H_
#define KALDI_DECODER_FASTER_DECODER_COUPLED_H_

#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc
#include <fstream> 
#include <map>

namespace kaldi {

struct FasterDecoderCoupledOptions {
  BaseFloat beam_1;
  BaseFloat beam_2;
  int ngram_1;
  int ngram_2;
  BaseFloat interpolation_1;
  BaseFloat interpolation_2;
  std::string coupled_lm_filename_1;
  std::string coupled_lm_filename_2;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta_1;
  BaseFloat beam_delta_2;
  BaseFloat hash_ratio;
  FasterDecoderCoupledOptions(): beam_1(16.0), beam_2(16.0),
	                  ngram_1(3), ngram_2(3),
			  interpolation_1(0.5),
			  interpolation_2(0.5),
	                  coupled_lm_filename_1("coupled-lm-1.arpa"),
			  coupled_lm_filename_2("coupled-lm-2.arpa"),
                          max_active(std::numeric_limits<int32>::max()),
                          min_active(20), // This decoder mostly used for
                                          // alignment, use small default.
                          beam_delta_1(0.5), beam_delta_2(0.5),
                          hash_ratio(2.0) { }
  void Register(OptionsItf *opts, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    opts->Register("beam_1", &beam_1, "Decoding beam for fst 1. Larger->slower, more accurate.");
    opts->Register("beam_2", &beam_2, "Decoding beam for fst 2. Larger->slower, more accurate.");
    opts->Register("interpolation_1", &interpolation_1, "Interpolation value for the coupled language model (1).");
    opts->Register("interpolation_2", &interpolation_2, "Interpolation value for the coupled language model (2).");
    opts->Register("ngram_1", &ngram_1, "Coupled language model 1 ngram size");
    opts->Register("ngram_2", &ngram_2, "Coupled language model 2 ngram size");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active,
                   "Decoder min active states (don't prune if #active less than this).");
    opts->Register("coupled-lm-1", &coupled_lm_filename_1, "Coupled language model file 1.");
    opts->Register("coupled-lm-2", &coupled_lm_filename_2, "Coupled language model file 2.");
    if (full) {
      opts->Register("beam-delta_1", &beam_delta_1,
                     "Increment used in decoder [obscure setting]");
      opts->Register("beam-delta_2", &beam_delta_2,
		     "Increment used in decoder [obscure setting]");
      opts->Register("hash-ratio", &hash_ratio,
                     "Setting used in decoder to control hash behavior");
    }
  }
};

class FasterDecoderCoupled {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  FasterDecoderCoupled(const fst::Fst<fst::StdArc> &fst_1, const fst::Fst<fst::StdArc> &fst_2,
                const FasterDecoderCoupledOptions &config);

  void SetOptions(const FasterDecoderCoupledOptions &config) { config_ = config; }

  ~FasterDecoderCoupled() { ClearToks_1(toks_1_.Clear()); ClearToks_2(toks_2_.Clear()); }

  void Decode(DecodableInterface *decodable_1, DecodableInterface *decodable_2);

  /// Returns true if a final state was active on the last frame.
  bool ReachedFinal_1() const;
  bool ReachedFinal_2() const;

  /// GetBestPath gets the decoding traceback. If "use_final_probs" is true
  /// AND we reached a final state, it limits itself to final states;
  /// otherwise it gets the most likely token not taking into account
  /// final-probs. Returns true if the output best path was not the empty
  /// FST (will only return false in unusual circumstances where
  /// no tokens survived).
  bool GetBestPath_1(fst::MutableFst<LatticeArc> *fst_out,
                   bool use_final_probs = true);
  bool GetBestPath_2(fst::MutableFst<LatticeArc> *fst_out,
		   bool use_final_probs = true);

  /// As a new alternative to Decode(), you can call InitDecoding
  /// and then (possibly multiple times) AdvanceDecoding().
  void InitDecoding();

  std::map <std::tuple<int, int>, std::tuple<double, double>> coupled_lm_1_2_;
  std::map <std::tuple<int, int, int>, std::tuple<double, double>> coupled_lm_1_3_;
  std::map <std::tuple<int, int, int, int>, std::tuple<double, double>> coupled_lm_1_4_;
  std::map <std::tuple<int, int, int, int, int>, std::tuple<double, double>> coupled_lm_1_5_;
  std::map <std::tuple<int, int, int, int, int, int>, std::tuple<double, double>> coupled_lm_1_6_;
  std::map <std::tuple<int, int, int, int, int, int, int>, std::tuple<double, double>> coupled_lm_1_7_;
  std::map <std::tuple<int, int, int, int, int, int, int, int>, std::tuple<double, double>> coupled_lm_1_8_;
  std::map <std::tuple<int, int>, std::tuple<double, double>> coupled_lm_2_2_;
  std::map <std::tuple<int, int, int>, std::tuple<double, double>> coupled_lm_2_3_;
  std::map <std::tuple<int, int, int, int>, std::tuple<double, double>> coupled_lm_2_4_;
  std::map <std::tuple<int, int, int, int, int>, std::tuple<double, double>> coupled_lm_2_5_;
  std::map <std::tuple<int, int, int, int, int, int>, std::tuple<double, double>> coupled_lm_2_6_;
  std::map <std::tuple<int, int, int, int, int, int, int>, std::tuple<double, double>> coupled_lm_2_7_;
  std::map <std::tuple<int, int, int, int, int, int, int, int>, std::tuple<double, double>> coupled_lm_2_8_;

  double CoupledLanguageModel12(int plabel_1[], int olabel_2); 
  double CoupledLanguageModel12_2(int plabel_1[], int olabel_2);
  double CoupledLanguageModel12_3(int plabel_1[], int olabel_2);
  double CoupledLanguageModel12_4(int plabel_1[], int olabel_2);
  double CoupledLanguageModel12_5(int plabel_1[], int olabel_2);
  double CoupledLanguageModel12_6(int plabel_1[], int olabel_2);
  double CoupledLanguageModel12_7(int plabel_1[], int olabel_2);
  double CoupledLanguageModel12_8(int plabel_1[], int olabel_2);
  double CoupledLanguageModel21(int plabel_2[], int olabel_1); 
  double CoupledLanguageModel21_2(int plabel_2[], int olabel_1);
  double CoupledLanguageModel21_3(int plabel_2[], int olabel_1);
  double CoupledLanguageModel21_4(int plabel_2[], int olabel_1);
  double CoupledLanguageModel21_5(int plabel_2[], int olabel_1);
  double CoupledLanguageModel21_6(int plabel_2[], int olabel_1);
  double CoupledLanguageModel21_7(int plabel_2[], int olabel_1);
  double CoupledLanguageModel21_8(int plabel_2[], int olabel_1);

  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.
  void AdvanceDecoding(DecodableInterface *decodable_1, DecodableInterface *decodable_2,
                       int32 max_num_frames = -1);

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded_1() const { return num_frames_decoded_1_; };
  int32 NumFramesDecoded_2() const { return num_frames_decoded_2_; };

 protected:

  class Token {
   public:
    Arc arc_; // contains only the graph part of the cost;
    // we can work out the acoustic part from difference between
    // "cost_" and prev->cost_.
    Token *prev_;
    int32 ref_count_;
    // if you are looking for weight_ here, it was removed and now we just have
    // cost_, which corresponds to ConvertToCost(weight_).
    double cost_;
    inline Token(const Arc &arc, BaseFloat ac_cost, Token *prev):
        arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight.Value() + ac_cost;
      } else {
        cost_ = arc.weight.Value() + ac_cost;
      }
    }
    inline Token(const Arc &arc, Token *prev):
        arc_(arc), prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight.Value();
      } else {
        cost_ = arc.weight.Value();
      }
    }
    inline bool operator < (const Token &other) {
      return cost_ > other.cost_;
    }

    inline static void TokenDelete(Token *tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == NULL) return;
        else tok = prev;
      }
#ifdef KALDI_PARANOID
      KALDI_ASSERT(tok->ref_count_ > 0);
#endif
    }
  };
  typedef HashList<StateId, Token*>::Elem Elem;


  /// Gets the weight cutoff.  Also counts the active tokens.
  double GetCutoff(Elem *list_head, size_t *tok_count,
                   BaseFloat *adaptive_beam, Elem **best_elem, int select_beam);

  void PossiblyResizeHash_1(size_t num_toks);
  void PossiblyResizeHash_2(size_t num_toks);

  // ProcessEmitting returns the likelihood cutoff used.
  // It decodes the frame num_frames_decoded_ of the decodable object
  // and then increments num_frames_decoded_
  std::pair<double, double> ProcessEmitting(DecodableInterface *decodable_1, DecodableInterface *decodable_2);

  // TODO: first time we go through this, could avoid using the queue.
  void ProcessNonemitting(double cutoff_1, double cutoff_2);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<StateId, Token*> toks_1_;
  HashList<StateId, Token*> toks_2_;
  const fst::Fst<fst::StdArc> &fst_1_;
  const fst::Fst<fst::StdArc> &fst_2_;
  FasterDecoderCoupledOptions config_;
  std::vector<const Elem* > queue_1_;  // temp variable used in ProcessNonemitting,
  std::vector<const Elem* > queue_2_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_1_;
  int32 num_frames_decoded_2_;

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks_1(Elem *list);
  void ClearToks_2(Elem *list);

  KALDI_DISALLOW_COPY_AND_ASSIGN(FasterDecoderCoupled);
};


} // end namespace kaldi.


#endif
