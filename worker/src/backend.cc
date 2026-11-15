#include "shardline/backend.h"

#include <algorithm>
#include <cctype>

namespace shardline {

std::string model_path_for_digest(const std::string& model_dir,
                                  const std::string& sha256) {
  bool valid = sha256.size() == 64 &&
               std::all_of(sha256.begin(), sha256.end(), [](unsigned char c) {
                 return std::isdigit(c) || (c >= 'a' && c <= 'f');
               });
  if (!valid) {
    throw BackendError(ErrorKind::kInvalidArgument,
                       "a model is named by its SHA-256: 64 lowercase hex digits");
  }
  std::string dir = model_dir;
  if (!dir.empty() && dir.back() != '/') {
    dir += '/';
  }
  return dir + sha256 + ".gguf";
}

#if !defined(SHARDLINE_HAVE_GGML)
std::unique_ptr<Backend> load_ggml_backend(const ShardSpec&) {
  throw BackendError(ErrorKind::kUnimplemented,
                     "this worker was built without ggml (SHARDLINE_GGML=OFF)");
}
#endif

}  // namespace shardline
