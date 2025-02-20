/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <fstream>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

namespace visnav {
bool compare(std::pair<FrameCamId, WordValue> l,
             std::pair<FrameCamId, WordValue> r) {
  return l.second < r.second;
}
class BowDatabase {
public:
  BowDatabase() {}

  inline void insert(const FrameCamId &fcid, const BowVector &bow_vector) {
    // TODO SHEET 3: add a bow_vector that corresponds to frame fcid to the
    // inverted index. You can assume the image hasn't been added before.

    // inverted_index

    for (size_t i = 0; i < bow_vector.size(); i++) {
      WordId wid = bow_vector[i].first;
      WordValue wval = bow_vector[i].second;

      if (inverted_index.count(wid) == 0) {
        tbb::concurrent_vector<std::pair<FrameCamId, WordValue>> v;
        v.push_back(std::make_pair(fcid, wval));
        inverted_index.insert(std::make_pair(wid, v));
      } else {
        inverted_index[wid].push_back(std::make_pair(fcid, wval));
      }
    }

    UNUSED(fcid);
    UNUSED(bow_vector);
  }

  inline void query(const BowVector &bow_vector, size_t num_results,
                    BowQueryResult &results) const {
    // TODO SHEET 3: find num_results closest matches to the bow_vector in the
    // inverted index. Hint: for good query performance use std::unordered_map
    // to accumulate scores and std::partial_sort for getting the closest
    // results. You should use L1 difference as the distance measure. You can
    // assume that BoW descripors are L1 normalized.
    std::unordered_map<FrameCamId, WordValue> score_accum;
    WordValue score;
    for (size_t i = 0; i < bow_vector.size(); i++) {
      WordId wid = bow_vector[i].first;
      WordValue wval = bow_vector[i].second;
      if (inverted_index.find(wid) == inverted_index.end())
        continue;

      for (size_t j = 0; j < inverted_index.at(wid).size(); j++) {
        FrameCamId fid = inverted_index.at(wid)[j].first;
        WordValue fwval = inverted_index.at(wid)[j].second;
        score = abs(fwval - wval) - abs(fwval) - abs(wval);

        if (score_accum[fid] == 0) {
          score_accum[fid] = score + 2;
        } else {
          score_accum[fid] += score;
        }
      }
    }
    if (score_accum.size() < num_results) {
      num_results = score_accum.size();
    }

    std::vector<std::pair<FrameCamId, WordValue>> f_score_vec(
        score_accum.begin(), score_accum.end());
    // size_t size_score = f_score_vec.size();
    // if (size_score < num_results) {
    //   num_results = size_score;
    // }
    std::sort(f_score_vec.begin(), f_score_vec.end(), compare);
    unsigned int i = 0;
    for (const auto &f_score : f_score_vec) {
      if (i < num_results) {
        results.push_back(f_score);

      } else {
        break;
      }
      i++;
    }

    UNUSED(bow_vector);
    UNUSED(num_results);
    UNUSED(results);
  }

  void clear() { inverted_index.clear(); }

  void save(const std::string &out_path) {
    BowDBInverseIndex state;
    for (const auto &kv : inverted_index) {
      for (const auto &a : kv.second) {
        state[kv.first].emplace_back(a);
      }
    }
    std::ofstream os;
    os.open(out_path, std::ios::binary);
    cereal::JSONOutputArchive archive(os);
    archive(state);
  }

  void load(const std::string &in_path) {
    BowDBInverseIndex inverseIndexLoaded;
    {
      std::ifstream os(in_path, std::ios::binary);
      cereal::JSONInputArchive archive(os);
      archive(inverseIndexLoaded);
    }
    for (const auto &kv : inverseIndexLoaded) {
      for (const auto &a : kv.second) {
        inverted_index[kv.first].emplace_back(a);
      }
    }
  }

  const BowDBInverseIndexConcurrent &getInvertedIndex() {
    return inverted_index;
  }

protected:
  BowDBInverseIndexConcurrent inverted_index;
};

} // namespace visnav
