#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

typedef int64_t int64;

class TokenizerBase {
protected:
  std::map<std::string, int64> tokenizer_token2idx;

public:
  virtual bool load_tokenize(std::string vocab_path) = 0;
  virtual void encode_text(std::string text, std::vector<int64> &idx) = 0;
  std::map<int64, std::string> tokenizer_idx2token;
};

class TokenizerClip : public TokenizerBase {
protected:
  std::vector<std::string> stringSplit(const std::string &str, char delim) {
    std::vector<std::string> elems;
    auto lastPos = str.find_first_not_of(delim, 0);
    auto pos = str.find_first_of(delim, lastPos);
    while (pos != std::string::npos || lastPos != std::string::npos) {
      elems.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delim, pos);
      pos = str.find_first_of(delim, lastPos);
    }
    return elems;
  }

  void tokenize(std::string token, std::vector<int64> &idx) {
    idx.push_back(101);
    {
      std::vector<std::string> tokens = stringSplit(token, ' ');
      for (auto t : tokens) {
        idx.push_back(tokenizer_token2idx[t]);
      }
    }
    idx.push_back(102);

    // memset(feat, 0, sizeof(CLIP_TEXT_FEATURE_T));
    // memcpy(feat->feature, idx.data(), idx.size() * sizeof(int));
  }

public:
  bool load_tokenize(std::string vocab_path) override {
    std::ifstream infile;
    infile.open(vocab_path.data());
    if (!infile.good()) {
      return false;
    }

    std::string s;
    int idx = 0;
    while (getline(infile, s)) {
      tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
      tokenizer_idx2token.insert(std::pair<int, std::string>(idx, s));
      idx++;
    }
    infile.close();
    return true;
  }

  void encode_text(std::string text, std::vector<int64> &idx) override {
    idx.clear();
    return tokenize(text, idx);
  }
};
