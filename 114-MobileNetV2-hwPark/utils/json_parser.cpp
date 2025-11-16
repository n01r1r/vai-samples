#include "utils/json_parser.h"

namespace mobilenet {

JsonParser::JsonParser() : loaded_(false) {
}

bool JsonParser::loadFromFile(const std::string& filePath) {
    // TODO: Implement JSON loading
    loaded_ = true;
    return true;
}

std::vector<float> JsonParser::parseArray(const std::string& key) {
    // TODO: Implement array parsing
    return {};
}

} // namespace mobilenet

