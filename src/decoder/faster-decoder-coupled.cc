// decoder/faster-decoder.cc

// Copyright 2009-2011 Microsoft Corporation
//           2012-2013 Johns Hopkins University (author: Daniel Povey)

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

#include "decoder/faster-decoder-coupled.h"

namespace kaldi {


FasterDecoderCoupled::FasterDecoderCoupled(const fst::Fst<fst::StdArc> &fst_1, const fst::Fst<fst::StdArc> &fst_2,
                             const FasterDecoderCoupledOptions &opts):
    fst_1_(fst_1), fst_2_(fst_2), config_(opts), num_frames_decoded_1_(-1), num_frames_decoded_2_(-1) {
  KALDI_ASSERT(config_.hash_ratio >= 1.0);  // less doesn't make much sense.
  KALDI_ASSERT(config_.max_active > 1);
  KALDI_ASSERT(config_.min_active >= 0 && config_.min_active < config_.max_active);
  toks_1_.SetSize(1000);  // just so on the first frame we do something reasonable.
  toks_2_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


void FasterDecoderCoupled::InitDecoding() {
  // clean up from last time:
  ClearToks_1(toks_1_.Clear());
  ClearToks_2(toks_2_.Clear());
  StateId start_state_1 = fst_1_.Start();
  StateId start_state_2 = fst_2_.Start();
  KALDI_ASSERT(start_state_1 != fst::kNoStateId);
  KALDI_ASSERT(start_state_2 != fst::kNoStateId);
  Arc dummy_arc_1(0, 0, Weight::One(), start_state_1);
  Arc dummy_arc_2(0, 0, Weight::One(), start_state_2);
  toks_1_.Insert(start_state_1, new Token(dummy_arc_1, NULL));
  toks_2_.Insert(start_state_2, new Token(dummy_arc_2, NULL));
  ProcessNonemitting(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  num_frames_decoded_1_ = 0;
  num_frames_decoded_2_ = 0;
  // Parse the coupled-lm files in integer format
  // The integer 0 cannot represent a symbol in the lm
  coupled_lm_1_2_.clear();
  coupled_lm_1_3_.clear();
  coupled_lm_1_4_.clear();
  coupled_lm_1_5_.clear();
  coupled_lm_1_6_.clear();
  coupled_lm_1_7_.clear();
  coupled_lm_1_8_.clear();
  coupled_lm_2_2_.clear();
  coupled_lm_2_3_.clear();
  coupled_lm_2_4_.clear();
  coupled_lm_2_5_.clear();
  coupled_lm_2_6_.clear();
  coupled_lm_2_7_.clear();
  coupled_lm_2_8_.clear();
  int i1, i2, i3, i4, i5, i6, i7, i8;
  double d1, d2;
  std::ifstream infile(config_.coupled_lm_filename_1);
  //KALDI_LOG << "Reading file " << config_.coupled_lm_filename_1;
  std::ifstream infile_2(config_.coupled_lm_filename_2);
  //KALDI_LOG << "Reading file " << config_.coupled_lm_filename_2;
  switch (config_.ngram_1) {
    case 2:
      while (infile >> i1 >> i2 >> d1 >> d2) {
        coupled_lm_1_2_[std::make_tuple(i1, i2)] = std::make_tuple(d1, d2);
      }
      break;
    case 3:
      while (infile >> i1 >> i2 >> i3 >> d1 >> d2) {
        coupled_lm_1_3_[std::make_tuple(i1, i2, i3)] = std::make_tuple(d1, d2);
      }
      break;
    case 4:
      while (infile >> i1 >> i2 >> i3 >> i4 >> d1 >> d2) {
        coupled_lm_1_4_[std::make_tuple(i1, i2, i3, i4)] = std::make_tuple(d1, d2);
      }
      break;
    case 5:
      while (infile >> i1 >> i2 >> i3 >> i4 >> i5 >> d1 >> d2) {
        coupled_lm_1_5_[std::make_tuple(i1, i2, i3, i4, i5)] = std::make_tuple(d1, d2);
      }
      break;
    case 6:
      while (infile >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> d1 >> d2) {
        coupled_lm_1_6_[std::make_tuple(i1, i2, i3, i4, i5, i6)] = std::make_tuple(d1, d2);
      }
      break;
    case 7:
      while (infile >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> d1 >> d2) {
        coupled_lm_1_7_[std::make_tuple(i1, i2, i3, i4, i5, i6, i7)] = std::make_tuple(d1, d2);
      }
      break;
    case 8:
      while (infile >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> d1 >> d2) {
        coupled_lm_1_8_[std::make_tuple(i1, i2, i3, i4, i5, i6, i7, i8)] = std::make_tuple(d1, d2);
      }
      break;
  }
  switch (config_.ngram_2) {
    case 2:
      while (infile_2 >> i1 >> i2 >> d1 >> d2) {
        coupled_lm_2_2_[std::make_tuple(i1, i2)] = std::make_tuple(d1, d2);
      }
      break;
    case 3:
      while (infile_2 >> i1 >> i2 >> i3 >> d1 >> d2) {
        coupled_lm_2_3_[std::make_tuple(i1, i2, i3)] = std::make_tuple(d1, d2);
      }
      break;
    case 4:
      while (infile_2 >> i1 >> i2 >> i3 >> i4 >> d1 >> d2) {
        coupled_lm_2_4_[std::make_tuple(i1, i2, i3, i4)] = std::make_tuple(d1, d2);
      }
      break;
    case 5:
      while (infile_2 >> i1 >> i2 >> i3 >> i4 >> i5 >> d1 >> d2) {
        coupled_lm_2_5_[std::make_tuple(i1, i2, i3, i4, i5)] = std::make_tuple(d1, d2);
      }
      break;
    case 6:
      while (infile_2 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> d1 >> d2) {
        coupled_lm_2_6_[std::make_tuple(i1, i2, i3, i4, i5, i6)] = std::make_tuple(d1, d2);
      }
      break;
    case 7:
      while (infile_2 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> d1 >> d2) {
        coupled_lm_2_7_[std::make_tuple(i1, i2, i3, i4, i5, i6, i7)] = std::make_tuple(d1, d2);
      }
      break;
    case 8:
      while (infile_2 >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> i8 >> d1 >> d2) {
        coupled_lm_2_8_[std::make_tuple(i1, i2, i3, i4, i5, i6, i7, i8)] = std::make_tuple(d1, d2);
      }
      break;
  }
}

void FasterDecoderCoupled::Decode(DecodableInterface *decodable_1, DecodableInterface *decodable_2) {
  InitDecoding();
  AdvanceDecoding(decodable_1, decodable_2);
}

void FasterDecoderCoupled::AdvanceDecoding(DecodableInterface *decodable_1, DecodableInterface *decodable_2,
                                      int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_1_ >= 0 &&
               "You must call InitDecoding() before AdvanceDecoding()");
  KALDI_ASSERT(num_frames_decoded_2_ >= 0 &&
	       "You must call InitDecoding() before AdvanceDecoding()");
  int32 num_frames_ready_1 = decodable_1->NumFramesReady();
  int32 num_frames_ready_2 = decodable_2->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready_1 >= num_frames_decoded_1_);
  KALDI_ASSERT(num_frames_ready_2 >= num_frames_decoded_2_);
  KALDI_ASSERT(num_frames_ready_1 == num_frames_ready_2);
  KALDI_ASSERT(num_frames_decoded_1_ == num_frames_decoded_2_);
  int32 target_frames_decoded_1 = num_frames_ready_1;
  int32 target_frames_decoded_2 = num_frames_ready_2;
  if (max_num_frames >= 0) {
    target_frames_decoded_1 = std::min(target_frames_decoded_1,
                                     num_frames_decoded_1_ + max_num_frames);
    target_frames_decoded_2 = std::min(target_frames_decoded_2,
		                     num_frames_decoded_2_ + max_num_frames);
  }
  while (num_frames_decoded_1_ < target_frames_decoded_1 && num_frames_decoded_2_ < target_frames_decoded_2) {
    // note: ProcessEmitting() increments num_frames_decoded_
    std::pair<double, double> weight_cutoffs = ProcessEmitting(decodable_1, decodable_2);
    ProcessNonemitting(weight_cutoffs.first, weight_cutoffs.second);
  }
  // AÑADIR FAILSAFE
}


bool FasterDecoderCoupled::ReachedFinal_1() const {
  for (const Elem *e = toks_1_.GetList(); e != NULL; e = e->tail) {
    if (e->val->cost_ != std::numeric_limits<double>::infinity() &&
        fst_1_.Final(e->key) != Weight::Zero()) {
      return true;
    }
  }
  return false;
}

bool FasterDecoderCoupled::ReachedFinal_2() const {
  for (const Elem *e = toks_2_.GetList(); e != NULL; e = e->tail) {
    if (e->val->cost_ != std::numeric_limits<double>::infinity() &&
        fst_2_.Final(e->key) != Weight::Zero()) {
      return true;
    }
  }
  return false;
}

bool FasterDecoderCoupled::GetBestPath_1(fst::MutableFst<LatticeArc> *fst_out,
                                bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal_1();
  if (!is_final) {
    for (const Elem *e = toks_1_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double infinity =  std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    for (const Elem *e = toks_1_.GetList(); e != NULL; e = e->tail) {
      double this_cost = e->val->cost_ + fst_1_.Final(e->key).Value();
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_1_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs) {
    Weight final_weight = fst_1_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}

bool FasterDecoderCoupled::GetBestPath_2(fst::MutableFst<LatticeArc> *fst_out,
                                bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal_2();
  if (!is_final) {
    for (const Elem *e = toks_2_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double infinity =  std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    for (const Elem *e = toks_2_.GetList(); e != NULL; e = e->tail) {
      double this_cost = e->val->cost_ + fst_2_.Final(e->key).Value();
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_2_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs) {
    Weight final_weight = fst_2_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}

// Gets the weight cutoff.  Also counts the active tokens.
double FasterDecoderCoupled::GetCutoff(Elem *list_head, size_t *tok_count,
                                BaseFloat *adaptive_beam, Elem **best_elem, int select_beam) {
  double best_cost = std::numeric_limits<double>::infinity();
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->cost_;
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) { 
      if (select_beam == 1) {
        *adaptive_beam = config_.beam_1;
      }
      else {
	*adaptive_beam = config_.beam_2;
      }
    };
    if (select_beam == 1) {
      return best_cost + config_.beam_1;
    } else {
      return best_cost + config_.beam_2;
    }
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->cost_;
      tmp_array_.push_back(w);
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    double beam_cutoff = std::numeric_limits<double>::infinity(),
	   min_active_cutoff = std::numeric_limits<double>::infinity(),
           max_active_cutoff = std::numeric_limits<double>::infinity();
    if (select_beam == 1) {
      beam_cutoff = best_cost + config_.beam_1;
    } else {
      beam_cutoff = best_cost + config_.beam_2;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam) {
	if (select_beam == 1) {
          *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta_1;
	} else {
          *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta_2;
	}
      }
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_cost;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam) {
	if (select_beam == 1) {
          *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta_1;
	} else {
          *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta_2;
	}
      }
      return min_active_cutoff;
    } else {
      if (select_beam == 1) {
        *adaptive_beam = config_.beam_1;
      } else {
	*adaptive_beam = config_.beam_2;
      }
      return beam_cutoff;
    }
  }
}

void FasterDecoderCoupled::PossiblyResizeHash_1(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_1_.Size()) {
    toks_1_.SetSize(new_sz);
  }
}

void FasterDecoderCoupled::PossiblyResizeHash_2(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
		                      * config_.hash_ratio);
  if (new_sz > toks_2_.Size()) {
    toks_2_.SetSize(new_sz);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12(int clm_labels_1[], int olabel_2) {
  if (olabel_2 == 0) {
    return 0;
  }
  switch (config_.ngram_1) {
    case 2:
      return CoupledLanguageModel12_2(clm_labels_1, olabel_2);
      break;
    case 3:
      return CoupledLanguageModel12_3(clm_labels_1, olabel_2);
      break;
    case 4:
      return CoupledLanguageModel12_4(clm_labels_1, olabel_2);
      break;
    case 5:
      return CoupledLanguageModel12_5(clm_labels_1, olabel_2);
      break;
    case 6:
      return CoupledLanguageModel12_6(clm_labels_1, olabel_2);
      break;
    case 7:
      return CoupledLanguageModel12_7(clm_labels_1, olabel_2);
      break;
    case 8:
      return CoupledLanguageModel12_8(clm_labels_1, olabel_2);
      break;
  }
  return 9999;
}

double FasterDecoderCoupled::CoupledLanguageModel12_2(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_2_.find(std::make_tuple(clm_labels_1[0], olabel_2));
  if (it == coupled_lm_1_2_.end()) {
    it = coupled_lm_1_2_.find(std::make_tuple(-1, olabel_2));
    if (it == coupled_lm_1_2_.end()) {
        return 9999;
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12_3(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_3_.find(std::make_tuple(clm_labels_1[0], clm_labels_1[1], olabel_2));
  if (it == coupled_lm_1_3_.end()) {
    it = coupled_lm_1_3_.find(std::make_tuple(-1, clm_labels_1[1], olabel_2));
    if (it == coupled_lm_1_3_.end()) {
      it = coupled_lm_1_3_.find(std::make_tuple(-1, -1, olabel_2));
      if (it == coupled_lm_1_3_.end()) {
        return 9999;
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12_4(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_4_.find(std::make_tuple(clm_labels_1[0], clm_labels_1[1], clm_labels_1[2], olabel_2));
  if (it == coupled_lm_1_4_.end()) {
    it = coupled_lm_1_4_.find(std::make_tuple(-1, clm_labels_1[1], clm_labels_1[2], olabel_2));
    if (it == coupled_lm_1_4_.end()) {
      it = coupled_lm_1_4_.find(std::make_tuple(-1, -1, clm_labels_1[2], olabel_2));
      if (it == coupled_lm_1_4_.end()) {
        it = coupled_lm_1_4_.find(std::make_tuple(-1, -1, -1, olabel_2));
        if (it == coupled_lm_1_4_.end()) {
          return 9999;
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12_5(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_5_.find(std::make_tuple(clm_labels_1[0], clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], olabel_2));
  if (it == coupled_lm_1_5_.end()) {
    it = coupled_lm_1_5_.find(std::make_tuple(-1, clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], olabel_2));
    if (it == coupled_lm_1_5_.end()) {
      it = coupled_lm_1_5_.find(std::make_tuple(-1, -1, clm_labels_1[2], clm_labels_1[3], olabel_2));
      if (it == coupled_lm_1_5_.end()) {
        it = coupled_lm_1_5_.find(std::make_tuple(-1, -1, -1, clm_labels_1[3], olabel_2));
        if (it == coupled_lm_1_5_.end()) {
          it = coupled_lm_1_5_.find(std::make_tuple(-1, -1, -1, -1, olabel_2));
          if (it == coupled_lm_1_5_.end()) {
            return 9999;
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12_6(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_6_.find(std::make_tuple(clm_labels_1[0], clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], olabel_2));
  if (it == coupled_lm_1_6_.end()) {
    it = coupled_lm_1_6_.find(std::make_tuple(-1, clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], olabel_2));
    if (it == coupled_lm_1_6_.end()) {
      it = coupled_lm_1_6_.find(std::make_tuple(-1, -1, clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], olabel_2));
      if (it == coupled_lm_1_6_.end()) {
        it = coupled_lm_1_6_.find(std::make_tuple(-1, -1, -1, clm_labels_1[3], clm_labels_1[4], olabel_2));
        if (it == coupled_lm_1_6_.end()) {
          it = coupled_lm_1_6_.find(std::make_tuple(-1, -1, -1, -1, clm_labels_1[4], olabel_2));
          if (it == coupled_lm_1_6_.end()) {
            it = coupled_lm_1_6_.find(std::make_tuple(-1, -1, -1, -1, -1, olabel_2));
            if (it == coupled_lm_1_6_.end()) {
              return 9999;
            } else {
              return - std::get<0>(it->second) - std::get<1>(it->second);
            }
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12_7(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_7_.find(std::make_tuple(clm_labels_1[0], clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], olabel_2));
  if (it == coupled_lm_1_7_.end()) {
    it = coupled_lm_1_7_.find(std::make_tuple(-1, clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], olabel_2));
    if (it == coupled_lm_1_7_.end()) {
      it = coupled_lm_1_7_.find(std::make_tuple(-1, -1, clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], olabel_2));
      if (it == coupled_lm_1_7_.end()) {
        it = coupled_lm_1_7_.find(std::make_tuple(-1, -1, -1, clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], olabel_2));
        if (it == coupled_lm_1_7_.end()) {
          it = coupled_lm_1_7_.find(std::make_tuple(-1, -1, -1, -1, clm_labels_1[4], clm_labels_1[5], olabel_2));
          if (it == coupled_lm_1_7_.end()) {
            it = coupled_lm_1_7_.find(std::make_tuple(-1, -1, -1, -1, -1, clm_labels_1[5], olabel_2));
            if (it == coupled_lm_1_7_.end()) {
              it = coupled_lm_1_7_.find(std::make_tuple(-1, -1, -1, -1, -1, -1, olabel_2));
              if (it == coupled_lm_1_7_.end()) {
                return 9999;
              } else {
                return - std::get<0>(it->second) - std::get<1>(it->second);
              }
            } else {
              return - std::get<0>(it->second) - std::get<1>(it->second);
            }
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel12_8(int clm_labels_1[], int olabel_2) {
  auto it = coupled_lm_1_8_.find(std::make_tuple(clm_labels_1[0], clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], clm_labels_1[6], olabel_2));
  if (it == coupled_lm_1_8_.end()) {
    it = coupled_lm_1_8_.find(std::make_tuple(-1, clm_labels_1[1], clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], clm_labels_1[6], olabel_2));
    if (it == coupled_lm_1_8_.end()) {
      it = coupled_lm_1_8_.find(std::make_tuple(-1, -1, clm_labels_1[2], clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], clm_labels_1[6], olabel_2));
      if (it == coupled_lm_1_8_.end()) {
        it = coupled_lm_1_8_.find(std::make_tuple(-1, -1, -1, clm_labels_1[3], clm_labels_1[4], clm_labels_1[5], clm_labels_1[6], olabel_2));
        if (it == coupled_lm_1_8_.end()) {
          it = coupled_lm_1_8_.find(std::make_tuple(-1, -1, -1, -1, clm_labels_1[4], clm_labels_1[5], clm_labels_1[6], olabel_2));
          if (it == coupled_lm_1_8_.end()) {
            it = coupled_lm_1_8_.find(std::make_tuple(-1, -1, -1, -1, -1, clm_labels_1[5], clm_labels_1[6], olabel_2));
            if (it == coupled_lm_1_8_.end()) {
              it = coupled_lm_1_8_.find(std::make_tuple(-1, -1, -1, -1, -1, -1, clm_labels_1[6], olabel_2));
              if (it == coupled_lm_1_8_.end()) {
                it = coupled_lm_1_8_.find(std::make_tuple(-1, -1, -1, -1, -1, -1, -1, olabel_2));
                if (it == coupled_lm_1_8_.end()) {
                  return 9999;
                } else {
                  return - std::get<0>(it->second) - std::get<1>(it->second);
                }
              } else {
                return - std::get<0>(it->second) - std::get<1>(it->second);
              }
            } else {
              return - std::get<0>(it->second) - std::get<1>(it->second);
            }
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21(int clm_labels_2[], int olabel_1) {
  if (olabel_1 == 0) {
    return 0;
  }
  switch (config_.ngram_2) {
    case 2:
      return CoupledLanguageModel21_2(clm_labels_2, olabel_1);
      break;
    case 3:
      return CoupledLanguageModel21_3(clm_labels_2, olabel_1);
      break;
    case 4:
      return CoupledLanguageModel21_4(clm_labels_2, olabel_1);
      break;
    case 5:
      return CoupledLanguageModel21_5(clm_labels_2, olabel_1);
      break;
    case 6:
      return CoupledLanguageModel21_6(clm_labels_2, olabel_1);
      break;
    case 7:
      return CoupledLanguageModel21_7(clm_labels_2, olabel_1);
      break;
    case 8:
      return CoupledLanguageModel21_8(clm_labels_2, olabel_1);
      break;
  }
  return 9999;
}

double FasterDecoderCoupled::CoupledLanguageModel21_2(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_2_.find(std::make_tuple(clm_labels_2[0], olabel_1));
  if (it == coupled_lm_2_2_.end()) {
    it = coupled_lm_2_2_.find(std::make_tuple(-1, olabel_1));
    if (it == coupled_lm_2_2_.end()) {
      return 9999;
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21_3(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_3_.find(std::make_tuple(clm_labels_2[0], clm_labels_2[1], olabel_1));
  if (it == coupled_lm_2_3_.end()) {
    it = coupled_lm_2_3_.find(std::make_tuple(-1, clm_labels_2[1], olabel_1));
    if (it == coupled_lm_2_3_.end()) {
      it = coupled_lm_2_3_.find(std::make_tuple(-1, -1, olabel_1));
      if (it == coupled_lm_2_3_.end()) {
        return 9999;
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21_4(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_4_.find(std::make_tuple(clm_labels_2[0], clm_labels_2[1], clm_labels_2[2], olabel_1));
  if (it == coupled_lm_2_4_.end()) {
    it = coupled_lm_2_4_.find(std::make_tuple(-1, clm_labels_2[1], clm_labels_2[2], olabel_1));
    if (it == coupled_lm_2_4_.end()) {
      it = coupled_lm_2_4_.find(std::make_tuple(-1, -1, clm_labels_2[2], olabel_1));
      if (it == coupled_lm_2_4_.end()) {
        it = coupled_lm_2_4_.find(std::make_tuple(-1, -1, -1, olabel_1));
        if (it == coupled_lm_2_4_.end()) {
          return 9999;
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21_5(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_5_.find(std::make_tuple(clm_labels_2[0], clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], olabel_1));
  if (it == coupled_lm_2_5_.end()) {
    it = coupled_lm_2_5_.find(std::make_tuple(-1, clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], olabel_1));
    if (it == coupled_lm_2_5_.end()) {
      it = coupled_lm_2_5_.find(std::make_tuple(-1, -1, clm_labels_2[2], clm_labels_2[3], olabel_1));
      if (it == coupled_lm_2_5_.end()) {
        it = coupled_lm_2_5_.find(std::make_tuple(-1, -1, -1, clm_labels_2[3], olabel_1));
        if (it == coupled_lm_2_5_.end()) {
          it = coupled_lm_2_5_.find(std::make_tuple(-1, -1, -1, -1, olabel_1));
          if (it == coupled_lm_2_5_.end()) {
            return 9999;
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21_6(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_6_.find(std::make_tuple(clm_labels_2[0], clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], olabel_1));
  if (it == coupled_lm_2_6_.end()) {
    it = coupled_lm_2_6_.find(std::make_tuple(-1, clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], olabel_1));
    if (it == coupled_lm_2_6_.end()) {
      it = coupled_lm_2_6_.find(std::make_tuple(-1, -1, clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], olabel_1));
      if (it == coupled_lm_2_6_.end()) {
        it = coupled_lm_2_6_.find(std::make_tuple(-1, -1, -1, clm_labels_2[3], clm_labels_2[4], olabel_1));
        if (it == coupled_lm_2_6_.end()) {
          it = coupled_lm_2_6_.find(std::make_tuple(-1, -1, -1, -1, clm_labels_2[4], olabel_1));
          if (it == coupled_lm_2_6_.end()) {
            it = coupled_lm_2_6_.find(std::make_tuple(-1, -1, -1, -1, -1, olabel_1));
            if (it == coupled_lm_2_6_.end()) {
              return 9999;
            } else {
              return - std::get<0>(it->second) - std::get<1>(it->second);
            }
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21_7(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_7_.find(std::make_tuple(clm_labels_2[0], clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], olabel_1));
  if (it == coupled_lm_2_7_.end()) {
    it = coupled_lm_2_7_.find(std::make_tuple(-1, clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], olabel_1));
    if (it == coupled_lm_2_7_.end()) {
      it = coupled_lm_2_7_.find(std::make_tuple(-1, -1, clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], olabel_1));
      if (it == coupled_lm_2_7_.end()) {
        it = coupled_lm_2_7_.find(std::make_tuple(-1, -1, -1, clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], olabel_1));
        if (it == coupled_lm_2_7_.end()) {
          it = coupled_lm_2_7_.find(std::make_tuple(-1, -1, -1, -1, clm_labels_2[4], clm_labels_2[5], olabel_1));
          if (it == coupled_lm_2_7_.end()) {
            it = coupled_lm_2_7_.find(std::make_tuple(-1, -1, -1, -1, -1, clm_labels_2[5], olabel_1));
            if (it == coupled_lm_2_7_.end()) {
              it = coupled_lm_2_7_.find(std::make_tuple(-1, -1, -1, -1, -1, -1, olabel_1));
              if (it == coupled_lm_2_7_.end()) {
                return 9999;
              } else {
                return - std::get<0>(it->second) - std::get<1>(it->second);
              }
            } else {
              return - std::get<0>(it->second) - std::get<1>(it->second);
            }
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

double FasterDecoderCoupled::CoupledLanguageModel21_8(int clm_labels_2[], int olabel_1) {
  auto it = coupled_lm_2_8_.find(std::make_tuple(clm_labels_2[0], clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], clm_labels_2[6], olabel_1));
  if (it == coupled_lm_2_8_.end()) {
    it = coupled_lm_2_8_.find(std::make_tuple(-1, clm_labels_2[1], clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], clm_labels_2[6], olabel_1));
    if (it == coupled_lm_2_8_.end()) {
      it = coupled_lm_2_8_.find(std::make_tuple(-1, -1, clm_labels_2[2], clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], clm_labels_2[6], olabel_1));
      if (it == coupled_lm_2_8_.end()) {
        it = coupled_lm_2_8_.find(std::make_tuple(-1, -1, -1, clm_labels_2[3], clm_labels_2[4], clm_labels_2[5], clm_labels_2[6], olabel_1));
        if (it == coupled_lm_2_8_.end()) {
          it = coupled_lm_2_8_.find(std::make_tuple(-1, -1, -1, -1, clm_labels_2[4], clm_labels_2[5], clm_labels_2[6], olabel_1));
          if (it == coupled_lm_2_8_.end()) {
            it = coupled_lm_2_8_.find(std::make_tuple(-1, -1, -1, -1, -1, clm_labels_2[5], clm_labels_2[6], olabel_1));
            if (it == coupled_lm_2_8_.end()) {
              it = coupled_lm_2_8_.find(std::make_tuple(-1, -1, -1, -1, -1, -1, clm_labels_2[6], olabel_1));
              if (it == coupled_lm_2_8_.end()) {
                it = coupled_lm_2_8_.find(std::make_tuple(-1, -1, -1, -1, -1, -1, -1, olabel_1));
                if (it == coupled_lm_2_8_.end()) {
                  return 9999;
                } else {
                  return - std::get<0>(it->second) - std::get<1>(it->second);
                }
              } else {
                return - std::get<0>(it->second) - std::get<1>(it->second);
              }
            } else {
              return - std::get<0>(it->second) - std::get<1>(it->second);
            }
          } else {
            return - std::get<0>(it->second) - std::get<1>(it->second);
          }
        } else {
          return - std::get<0>(it->second) - std::get<1>(it->second);
        }
      } else {
        return - std::get<0>(it->second) - std::get<1>(it->second);
      }
    } else {
      return - std::get<0>(it->second) - std::get<1>(it->second);
    }
  } else {
    return - std::get<0>(it->second) - std::get<1>(it->second);
  }
}

// ProcessEmitting returns the likelihood cutoff used.
std::pair<double, double> FasterDecoderCoupled::ProcessEmitting(DecodableInterface *decodable_1, DecodableInterface *decodable_2) {
  int32 frame_1 = num_frames_decoded_1_;
  int32 frame_2 = num_frames_decoded_2_;
  Elem *last_toks_1 = toks_1_.Clear();
  Elem *last_toks_2 = toks_2_.Clear();
  size_t tok_cnt_1;
  size_t tok_cnt_2;
  BaseFloat adaptive_beam_1;
  BaseFloat adaptive_beam_2;
  // Keep finding best elements just in case, not necessary anymore
  Elem *best_elem_1 = NULL;
  Elem *best_elem_2 = NULL;
  double weight_cutoff_1 = GetCutoff(last_toks_1, &tok_cnt_1,
                                   &adaptive_beam_1, &best_elem_1, 1);
  double weight_cutoff_2 = GetCutoff(last_toks_2, &tok_cnt_2,
		                   &adaptive_beam_2, &best_elem_2, 2);
  // Best elements to form pairs
  Token *best_next_elem_1 = NULL;
  Token *best_next_elem_2 = NULL;
  // double best_next_elem_ac_cost_1 = std::numeric_limits<double>::infinity();
  // double best_next_elem_ac_cost_2 = std::numeric_limits<double>::infinity();
  // double best_next_elem_arcw_1 = std::numeric_limits<double>::infinity();
  // double best_next_elem_arcw_2 = std::numeric_limits<double>::infinity();
  // int best_next_elem_olabel_1 = 0;
  // int best_next_elem_olabel_2 = 0;
  // int best_next_elem_plabel_1 = 0;
  // int best_next_elem_plabel_2 = 0;
  //int clm_labels_1 [2] = { 0, 0 };
  //int clm_labels_2 [2] = { 0, 0 };
  int clm_labels_1 [7] = { 0, 0, 0, 0, 0, 0, 0 };
  int clm_labels_2 [7] = { 0, 0, 0, 0, 0, 0, 0 };

  KALDI_VLOG(3) << tok_cnt_1 << " tokens active in 1.";
  KALDI_VLOG(3) << tok_cnt_2 << " tokens active in 2.";
  PossiblyResizeHash_1(tok_cnt_1);  // This makes sure the hash is always big enough.
  PossiblyResizeHash_2(tok_cnt_2);  // This makes sure the hash is always big enough.

  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  double next_weight_cutoff_1 = std::numeric_limits<double>::infinity();
  double next_weight_cutoff_2 = std::numeric_limits<double>::infinity();

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  // if (best_elem_1) {
  //   StateId state_1 = best_elem_1->key;
  //   Token *tok_1 = best_elem_1->val;
  //   for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_1_, state_1);
  //        !aiter.Done();
  //        aiter.Next()) {
  //     const Arc &arc = aiter.Value();
  //     if (arc.ilabel != 0) {  // we'd propagate..
  //       BaseFloat ac_cost = - decodable_1->LogLikelihood(frame_1, arc.ilabel);
  //       double new_weight = arc.weight.Value() + tok_1->cost_ + ac_cost;
  //       if (new_weight + adaptive_beam < next_weight_cutoff_1)
  //         next_weight_cutoff_1 = new_weight + adaptive_beam;
  //       if (new_weight < best_weight_cutoff_1)
  //         best_weight_cutoff_1 = new_weight
  //     }
  //   }
  // }
  // if (best_elem_2) {
  //   StateId state_2 = best_elem_2->key;
  //   Token *tok_2 = best_elem_2->val;
  //   for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_2_, state_2);
  //        !aiter.Done();
  //        aiter.Next()) {
  //     const Arc &arc = aiter.Value();
  //     if (arc.ilabel != 0) {  // we'd propagate..
  //       BaseFloat ac_cost = - decodable_2->LogLikelihood(frame_2, arc.ilabel);
  //       double new_weight = arc.weight.Value() + tok_2->cost_ + ac_cost;
  //       if (new_weight + adaptive_beam < next_weight_cutoff_2)
  //         next_weight_cutoff_2 = new_weight + adaptive_beam;
  //       if (new_weight < best_weight_cutoff_2)
  //         best_weight_cutoff_2 = new_weight
  //     }
  //   }
  // }
  
  // int32 n = 0, np = 0;

  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call TokenDelete
  // on each elem 'e' to let toks_ know we're done with them.
  
  // Find the best element on each sequence to form the pairs
  for (Elem *e_1 = last_toks_1; e_1 != NULL; e_1 = e_1->tail) {
    // Iterate over every element without deletion
    StateId state_1 = e_1->key;
    Token *tok_1 = e_1->val;
    if (tok_1->cost_ < weight_cutoff_1) {  // not pruned.
      KALDI_ASSERT(state_1 == tok_1->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_1_, state_1);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
	if (arc.ilabel != 0) {  // propagate..
	  BaseFloat ac_cost =  - decodable_1->LogLikelihood(frame_1, arc.ilabel);
	  double new_weight = arc.weight.Value() + tok_1->cost_ + ac_cost;
	  if (new_weight + adaptive_beam_1 < next_weight_cutoff_1) {
	    // Update the best element to form a pair
	    best_next_elem_1 = new Token(arc, ac_cost, tok_1);
	    // best_next_elem_ac_cost_1 = ac_cost;
	    // best_next_elem_olabel_1 = arc.olabel;
	    // best_next_elem_arcw_1 = arc.weight.Value();
	    next_weight_cutoff_1 = new_weight + adaptive_beam_1;
	  }
	}
      }
    }
  }
  // Get previous output symbols for language model estimation
  // for (Token *tok_1 = best_next_elem_1; tok_1 != NULL; tok_1 = tok_1->prev_) {
  //   if (tok_1 != best_next_elem_1) {
  //     best_next_elem_plabel_1 = tok_1->arc_.olabel;
  //     if (best_next_elem_plabel_1 != 0) {
  //       break;
  //     }
  //   }
  // }
  int iter = config_.ngram_1 - 2;
  for (Token *tok_1 = best_next_elem_1; tok_1 != NULL && iter >= 0; tok_1 = tok_1->prev_) {
    if (tok_1 != best_next_elem_1) {
      if (tok_1->arc_.olabel != 0) {
        clm_labels_1 [iter] = tok_1->arc_.olabel;
	iter--;
      }
    }
  }

  for (Elem *e_2 = last_toks_2; e_2 != NULL; e_2 = e_2->tail) {
    // Iterate over every element without deletion
    StateId state_2 = e_2->key;
    Token *tok_2 = e_2->val;
    if (tok_2->cost_ < weight_cutoff_2) {  // not pruned.
      KALDI_ASSERT(state_2 == tok_2->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_2_, state_2);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost =  - decodable_2->LogLikelihood(frame_2, arc.ilabel);
          double new_weight = arc.weight.Value() + tok_2->cost_ + ac_cost;
          if (new_weight + adaptive_beam_2 < next_weight_cutoff_2) {
	    // Update the best element to form a pair
            best_next_elem_2 = new Token(arc, ac_cost, tok_2);
	    // best_next_elem_ac_cost_2 = ac_cost;
	    // best_next_elem_olabel_2 = arc.olabel;
	    // best_next_elem_arcw_2 = arc.weight.Value();
	    next_weight_cutoff_2 = new_weight + adaptive_beam_2;
          }
        }
      }
    }
  }
  // Get previous output symbol for language model estimation
  // for (Token *tok_2 = best_next_elem_2; tok_2 != NULL; tok_2 = tok_2->prev_) {
  //   if (tok_2 != best_next_elem_2) {
  //     best_next_elem_plabel_2 = tok_2->arc_.olabel;
  //     if (best_next_elem_plabel_2 != 0) {
  //       break;
  //     }
  //   }
  // }
  iter = config_.ngram_2 - 2;
  for (Token *tok_2 = best_next_elem_2; tok_2 != NULL && iter >= 0; tok_2 = tok_2->prev_) {
    if (tok_2 != best_next_elem_2) {
      if (tok_2->arc_.olabel != 0) {
        clm_labels_2 [iter] = tok_2->arc_.olabel;
        iter--;
      }
    }
  }

  // Get next_weight_cutoff based on the pair of the best individual elements
  double next_weight_cutoff_pair_1 = next_weight_cutoff_1 + next_weight_cutoff_2; 
  double next_weight_cutoff_pair_2 = next_weight_cutoff_pair_1;
  
  // Get all pairs that are better than the cutoff
  for (Elem *e_1 = last_toks_1, *e_1_tail; e_1 != NULL; e_1 = e_1_tail) {  // loop this way
    // n++;
    // because we delete "e" as we go.
    StateId state_1 = e_1->key;
    Token *tok_1 = e_1->val;
    // int this_elem_plabel_1 = 0;
    // for (Token *tok_1_1 = tok_1; tok_1_1 != NULL; tok_1_1 = tok_1_1->prev_) {
    //   this_elem_plabel_1 = tok_1_1->arc_.olabel;
    //   if (this_elem_plabel_1 != 0) {
    //     break;
    //   }
    // }
    // ESTE WEIGHT CUTOFF PUEDE REQUERIR CAMBIO
    if (tok_1->cost_ < weight_cutoff_1) {  // not pruned.
      // np++;
      KALDI_ASSERT(state_1 == tok_1->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_1_, state_1);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost =  - decodable_1->LogLikelihood(frame_1, arc.ilabel)
		  + CoupledLanguageModel21(clm_labels_2, arc.olabel) * config_.interpolation_1;
	  //KALDI_LOG << "Dictionary result " << CoupledLanguageModel21(clm_labels_2, arc.olabel);
          // double new_weight = arc.weight.Value() + tok_1->cost_ + ac_cost;
	  double new_weight = tok_1->cost_ + arc.weight.Value() + ac_cost;
          // if (new_weight < next_weight_cutoff_1) {  // not pruned..
	  if (new_weight < next_weight_cutoff_pair_1) {  // not pruned..
	    // get accumulated cost for the new token
            Token *new_tok = new Token(arc, ac_cost, tok_1);
            Elem *e_found = toks_1_.Insert(arc.nextstate, new_tok);
            if (new_weight + adaptive_beam_1 + adaptive_beam_2 < next_weight_cutoff_pair_1)
              next_weight_cutoff_pair_1 = new_weight + adaptive_beam_1 + adaptive_beam_2;
            if (e_found->val != new_tok) {
              if (*(e_found->val) < *new_tok) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
    e_1_tail = e_1->tail;
    Token::TokenDelete(e_1->val);
    toks_1_.Delete(e_1);
  }
  for (Elem *e_2 = last_toks_2, *e_2_tail; e_2 != NULL; e_2 = e_2_tail) {  // loop this way
    StateId state_2 = e_2->key;
    Token *tok_2 = e_2->val;
    // int this_elem_plabel_2 = 0;
    // for (Token *tok_2_2 = tok_2; tok_2_2 != NULL; tok_2_2 = tok_2_2->prev_) {
    //   this_elem_plabel_2 = tok_2_2->arc_.olabel;
    //   if (this_elem_plabel_2 != 0) {
    //     break;
    //   }
    // }
    // ESTE WEIGHT CUTOFF PUEDE REQUERIR CAMBIO
    if (tok_2->cost_ < weight_cutoff_2) {  // not pruned.
      // np++;
      KALDI_ASSERT(state_2 == tok_2->arc_.nextstate);
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_2_, state_2);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost =  - decodable_2->LogLikelihood(frame_2, arc.ilabel)
		  + CoupledLanguageModel12(clm_labels_1, arc.olabel)  * config_.interpolation_2;
	  //KALDI_LOG << "Dictionary result " << CoupledLanguageModel12(clm_labels_1, arc.olabel);
          // double new_weight = arc.weight.Value() + tok_2->cost_ + ac_cost;
	  double new_weight = tok_2->cost_ + arc.weight.Value() + ac_cost;
          // if (new_weight < next_weight_cutoff_2) {  // not pruned..
	  if (new_weight < next_weight_cutoff_pair_2) {  // not pruned..
	    // get accumulated cost for the new token
            Token *new_tok = new Token(arc, ac_cost, tok_2);
            Elem *e_found = toks_2_.Insert(arc.nextstate, new_tok);
            if (new_weight + adaptive_beam_1 + adaptive_beam_2 < next_weight_cutoff_pair_2)
              next_weight_cutoff_pair_2 = new_weight + adaptive_beam_1 + adaptive_beam_2;
            if (e_found->val != new_tok) {
              if (*(e_found->val) < *new_tok) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
    e_2_tail = e_2->tail;
    Token::TokenDelete(e_2->val);
    toks_2_.Delete(e_2);
  }
  // AÑADIR FAILSAFE
  num_frames_decoded_1_++;
  num_frames_decoded_2_++;
  return std::make_pair(next_weight_cutoff_pair_1, next_weight_cutoff_pair_2);
}

// TODO: first time we go through this, could avoid using the queue.
void FasterDecoderCoupled::ProcessNonemitting(double cutoff_1, double cutoff_2) {
  // Processes nonemitting arcs for one frame.
  KALDI_ASSERT(queue_1_.empty());
  KALDI_ASSERT(queue_2_.empty());
  for (const Elem *e = toks_1_.GetList(); e != NULL;  e = e->tail)
    queue_1_.push_back(e);
  for (const Elem *e = toks_2_.GetList(); e != NULL;  e = e->tail)
    queue_2_.push_back(e);
  // CONTINUAR AQUI
  while (!queue_1_.empty()) {
    const Elem* e_1 = queue_1_.back();
    queue_1_.pop_back();
    StateId state_1 = e_1->key;
    Token *tok_1 = e_1->val;  // would segfault if state not
    // in toks_ but this can't happen.
    if (tok_1->cost_ > cutoff_1) { // Don't bother processing successors.
      continue;
    }
    KALDI_ASSERT(tok_1 != NULL && state_1 == tok_1->arc_.nextstate);
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_1_, state_1);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        Token *new_tok = new Token(arc, tok_1);
        if (new_tok->cost_ > cutoff_1) {  // prune
          Token::TokenDelete(new_tok);
        } else {
          Elem *e_found = toks_1_.Insert(arc.nextstate, new_tok);
          if (e_found->val == new_tok) {
            queue_1_.push_back(e_found);
          } else {
            if (*(e_found->val) < *new_tok) {
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_1_.push_back(e_found);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
  while (!queue_2_.empty()) {
    const Elem* e_2 = queue_2_.back();
    queue_2_.pop_back();
    StateId state_2 = e_2->key;
    Token *tok_2 = e_2->val;  // would segfault if state not
    // in toks_ but this can't happen.
    if (tok_2->cost_ > cutoff_2) { // Don't bother processing successors.
      continue;
    }
    KALDI_ASSERT(tok_2 != NULL && state_2 == tok_2->arc_.nextstate);
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(fst_2_, state_2);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        Token *new_tok = new Token(arc, tok_2);
        if (new_tok->cost_ > cutoff_2) {  // prune
          Token::TokenDelete(new_tok);
        } else {
          Elem *e_found = toks_2_.Insert(arc.nextstate, new_tok);
          if (e_found->val == new_tok) {
            queue_2_.push_back(e_found);
          } else {
            if (*(e_found->val) < *new_tok) {
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_2_.push_back(e_found);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

void FasterDecoderCoupled::ClearToks_1(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_1_.Delete(e);
  }
}

void FasterDecoderCoupled::ClearToks_2(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_2_.Delete(e);
  }
}

} // end namespace kaldi.
