// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_EVAL_ALPHA_ZERO_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_EVAL_ALPHA_ZERO_H_

#include "open_spiel/utils/json.h"
#include "open_spiel/utils/thread.h"

namespace open_spiel
{
  namespace algorithms
  {
    namespace torch_az_eval
    {

      struct AlphaZeroConfig
      {
        std::string game;
        std::string path;
        std::string graph_def;
        std::string nn_model;
        int nn_width;
        int nn_depth;
        std::string devices;

        bool explicit_learning;
        double learning_rate;
        double weight_decay;
        int train_batch_size;
        int inference_batch_size;
        int inference_threads;
        int inference_cache;
        int replay_buffer_size;
        int replay_buffer_reuse;
        int checkpoint_freq;
        int evaluation_window;

        double uct_c;
        int max_simulations;
        double policy_alpha;
        double policy_epsilon;
        double temperature;
        double temperature_drop;
        double cutoff_probability;
        double cutoff_value;

        int actors;
        int evaluators;
        int eval_levels;
        int max_steps;

        double pw_exp;
        double ar_factor;
        double ar_exp;
        bool use_ar;
        bool perturb;
        int jobnum;
        int num_games;

        json::Object
        ToJson() const
        {
          return json::Object({
              {"game", game},
              {"path", path},
              {"graph_def", graph_def},
              {"nn_model", nn_model},
              {"nn_width", nn_width},
              {"nn_depth", nn_depth},
              {"devices", devices},
              {"explicit_learning", explicit_learning},
              {"learning_rate", learning_rate},
              {"weight_decay", weight_decay},
              {"train_batch_size", train_batch_size},
              {"inference_batch_size", inference_batch_size},
              {"inference_threads", inference_threads},
              {"inference_cache", inference_cache},
              {"replay_buffer_size", replay_buffer_size},
              {"replay_buffer_reuse", replay_buffer_reuse},
              {"checkpoint_freq", checkpoint_freq},
              {"evaluation_window", evaluation_window},
              {"uct_c", uct_c},
              {"max_simulations", max_simulations},
              {"policy_alpha", policy_alpha},
              {"policy_epsilon", policy_epsilon},
              {"temperature", temperature},
              {"temperature_drop", temperature_drop},
              {"cutoff_probability", cutoff_probability},
              {"cutoff_value", cutoff_value},
              {"actors", actors},
              {"evaluators", evaluators},
              {"eval_levels", eval_levels},
              {"max_steps", max_steps},
              {"pw_exp", pw_exp},
              {"ar_factor", ar_factor},
              {"ar_exp", ar_exp},
              {"use_ar", use_ar},
              {"perturb", perturb},
              {"jobnum", jobnum},
              {"num_games", num_games},
          });
        }
      };

      bool AlphaZero(AlphaZeroConfig config, StopToken *stop);

    } // namespace torch_az
  }   // namespace algorithms
} // namespace open_spiel

#endif // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_TORCH_ALPHA_ZERO_H_
