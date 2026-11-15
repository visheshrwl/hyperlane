#include <string>

#include <gtest/gtest.h>

#include "shardline/backend.h"

namespace shardline {
namespace {

const std::string kDigest(64, 'a');

TEST(ModelPath, IsTheDigestInsideTheModelDirectory) {
  EXPECT_EQ(model_path_for_digest("/models", kDigest), "/models/" + kDigest + ".gguf");
  EXPECT_EQ(model_path_for_digest("/models/", kDigest), "/models/" + kDigest + ".gguf");
  EXPECT_EQ(model_path_for_digest("", kDigest), kDigest + ".gguf");
}

TEST(ModelPath, AcceptsAllHexDigits) {
  EXPECT_NO_THROW(model_path_for_digest("/m", "0123456789abcdef" + std::string(48, '0')));
}

TEST(ModelPath, RefusesAnythingThatIsNotADigest) {
  for (const std::string& bad : {
           std::string(),
           std::string(63, 'a'),
           std::string(65, 'a'),
           std::string(64, 'A'),
           std::string(64, 'g'),
           "../" + std::string(61, 'a'),
           std::string(61, 'a') + "/..",
           std::string(32, 'a') + "/" + std::string(31, 'a'),
           std::string(63, 'a') + std::string(1, '\0'),
       }) {
    try {
      model_path_for_digest("/models", bad);
      FAIL() << "accepted " << bad;
    } catch (const BackendError& error) {
      EXPECT_EQ(error.kind(), ErrorKind::kInvalidArgument);
    }
  }
}

}  // namespace
}  // namespace shardline
