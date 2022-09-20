// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc/dual_net/migraphx_dual_net.h"

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"

#include "migraphx/migraphx.hpp"

#include "wtf/macros.h"

namespace minigo {
namespace {

class MigraphxDualNet : public Model {
 public:
  MigraphxDualNet(const ModelDefinition& def,
                  const FeatureDescriptor& feature_desc);

  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

 private:
  // TODO(aayujain): Check if these are required.
  void Reserve(int capacity);
  migraphx::program prog;
  migraphx::file_options options;
  std::vector<float> inputs_;
  const std::string graph_path;
  int batch_capacity = 0;
  migraphx_shape_datatype_t input_type;
  migraphx::api::shape input_shape;
  bool warmup=true;
};

MigraphxDualNet::MigraphxDualNet(const ModelDefinition& def,
                                 const FeatureDescriptor& feature_desc)
    : Model(std::string(file::Stem(def.path)), feature_desc),
      graph_path(def.path) {
  // TODO(aayujain): code here
  // Opt1: receive compiled and quantized model. TF does that.
  //       That means BS will be fixed and quantize (Y/N) and send to GPU also.
  //       How to handle dynamic size?
  // Opt2: receive def. Compile, quantize, sendtoGPU set here.
  //       May be able to handle dynamic sizes here?
  // Switched from Opt2 in earlier implementation to Opt1 now.

  options.set_file_format("json");
  prog = migraphx::load_buffer(def.model_bytes, options);
  auto param_shapes = prog.get_parameter_shapes();
  // Hard-coding single input for now.
  input_type = param_shapes["pos_tensor"].type();
  input_shape = param_shapes["pos_tensor"];
}

void MigraphxDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                         std::vector<ModelOutput*>* outputs,
                         std::string* model_name) {
  // Warmup: run warmup once.
  // Running here instead of constructor because it needs to run on the same
  // thread as where inference will be run.
  if (warmup) {
    // NOTE: The batchsize here needs to be identical to FLAGS.trt_max_batch_size
    std::vector<uint8_t> tmp(128*13*19*19, 0);
    migraphx::program_parameters tmp_params;
    auto tmp_shapes = prog.get_parameter_shapes();
    for(auto&& name: tmp_shapes.names())
      tmp_params.add(name, migraphx::argument(tmp_shapes[name], tmp.data()));
    prog.eval(tmp_params);
    warmup=false;
  }

  Reserve(inputs.size());

  WTF_SCOPE("TfDualNet::Run: inputs, capacity", size_t, int)
  (inputs.size(), batch_capacity);
  MG_CHECK(inputs.size() == outputs->size());

  // TODO(aayujain): check if batch_capacity has to be changed to batch_size
  auto shape = feature_descriptor().GetInputShape(batch_capacity);

  // program_params being read twice.
  // TODO(aayujain): make class attribute to avoid repetition?
  auto param_shapes = prog.get_parameter_shapes();
  migraphx::program_parameters prog_params;

  if (input_type == migraphx_shape_float_type) {
    WTF_SCOPE("Features::SetFloat: inputs", int)(inputs.size());
    Tensor<float> features(shape, inputs_.data());
    feature_descriptor().set_floats(inputs, &features);

    for(auto&& name: param_shapes.names()) {
      prog_params.add(name, migraphx::argument(param_shapes[name], inputs_.data()));
    }
  }
  else {
    WTF_SCOPE("Features::SetBool: inputs", size_t)(inputs.size());
    static_assert(sizeof(bool) == sizeof(uint8_t), "bool must be 1 byte");
    Tensor<uint8_t> features(
        shape, reinterpret_cast<uint8_t*>(inputs_.data()));
    feature_descriptor().set_bytes(inputs, &features);

    for(auto&& name: param_shapes.names()) {
      prog_params.add(name, migraphx::argument(param_shapes[name], reinterpret_cast<uint8_t*>(inputs_.data())));
    }
  }

  // Run
  auto outputs_ = prog.eval(prog_params);

  Tensor<float> policy({batch_capacity, kNumMoves},
                       reinterpret_cast<float*>(outputs_[0].data()));
  Tensor<float> value({batch_capacity}, reinterpret_cast<float*>(outputs_[1].data()));
  {
    WTF_SCOPE("Model::GetOutputs: outputs", size_t)(outputs->size());
    Model::GetOutputs(inputs, policy, value, absl::MakeSpan(*outputs));
  }

  if (model_name != nullptr) {
    *model_name = graph_path;
  }
}

void MigraphxDualNet::Reserve(int capacity) {
  MG_CHECK(capacity > 0);
  if (capacity <= batch_capacity && capacity > 3 * batch_capacity / 4) {
    return;
  }

  inputs_.clear();

  // pos_tensor
  auto shape = feature_descriptor().GetInputShape(capacity);
  // TODO think about conditional resizing only if the shape changes.
  inputs_.resize(shape[0] * shape[1] * shape[2] * shape[3]);

  batch_capacity = capacity;
}

} // namespace

MigraphxDualNetFactory::MigraphxDualNetFactory(absl::string_view device) {
  // Place all models on the GPU by default, or if the user has explicitly
  // requested it.
  place_on_gpu_ = device.empty() || device == "gpu";
  if (!place_on_gpu_) {
    MG_CHECK(device == "cpu") << "Unrecognized device \"" << device << "\"";
  }
}

std::unique_ptr<Model> MigraphxDualNetFactory::NewModel(const ModelDefinition& def) {
  MG_CHECK(def.metadata.Get<std::string>("engine") == "migraphx");

  auto feature_desc =
      FeatureDescriptor::Create(def.metadata.Get<std::string>("input_features"),
                                def.metadata.Get<std::string>("input_layout"));

  return absl::make_unique<MigraphxDualNet>(def, feature_desc);
}

} // namespace minigo
