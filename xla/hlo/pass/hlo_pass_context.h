/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_PASS_HLO_PASS_CONTEXT_H_
#define XLA_HLO_PASS_HLO_PASS_CONTEXT_H_

#include <any>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xla {

// Type-indexed collection of non-owning pointers to the environment objects
// (e.g. se::StreamExecutor, se::DeviceDescription, mlir::MLIRContext) which
// HLO pass factories need to instantiate passes.
//
// Objects are looked up by the exact type they were registered with, e.g.
// Get<Base>() doesn't find an object registered as Derived (use
// Set<Base>(derived) to register it as Base). An object registered as
// non-const T is retrievable both as T and const T; one registered as const T
// only as const T.
//
// The objects must outlive the context.
class HloPassContext {
 public:
  // Returns AlreadyExists if an object of type T (const or not) is already
  // registered.
  template <typename T>
  absl::Status Set(T& value) {
    if (!objects_.try_emplace(typeid(const T*), &std::as_const(value)).second) {
      return absl::AlreadyExistsError(
          absl::StrCat("HloPassContext already has ", typeid(T).name()));
    }
    objects_.try_emplace(typeid(T*), &value);
    return absl::OkStatus();
  }

  // Rejects temporaries, which would dangle (e.g. Set<const T>(T{})).
  template <typename T>
  absl::Status Set(const T&&) = delete;

  // Returns NotFound if no object of type T is registered.
  template <typename T>
  absl::StatusOr<T&> Get() const {
    if (T* value = GetOrNull<T>()) {
      return *value;
    }
    return absl::NotFoundError(absl::StrCat("HloPassContext has no ",
                                            Has<const T>() ? "non-const " : "",
                                            typeid(T).name()));
  }

  template <typename T>
  T* GetOrNull() const {
    auto it = objects_.find(typeid(T*));
    return it != objects_.end() ? std::any_cast<T*>(it->second) : nullptr;
  }

  template <typename T>
  T& GetOrDie() const {
    T* value = GetOrNull<T>();
    CHECK(value != nullptr) << Get<T>().status();
    return *value;
  }

  template <typename T>
  bool Has() const {
    return objects_.contains(typeid(T*));
  }

 private:
  // Keyed by T* rather than T because typeid drops top-level constness.
  // Non-const T is stored under both const T* and T* so const lookups need no
  // fallback.
  absl::flat_hash_map<std::type_index, std::any> objects_;
};

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_CONTEXT_H_
