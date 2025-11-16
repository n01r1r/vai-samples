#ifndef MOBILENET_JSON_PARSER_H
#define MOBILENET_JSON_PARSER_H

#include <string>
#include <vector>

namespace mobilenet {

class JsonParser {
public:
    JsonParser();
    ~JsonParser() = default;
    
    bool loadFromFile(const std::string& filePath);
    std::vector<float> parseArray(const std::string& key);
    
private:
    bool loaded_;
};

} // namespace mobilenet

#endif // MOBILENET_JSON_PARSER_H

