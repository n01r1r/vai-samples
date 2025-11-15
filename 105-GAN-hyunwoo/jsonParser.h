#ifndef JSONPARSER_H
#define JSONPARSER_H


#include <vector>
#include <memory>
#include <string_view>

struct JsonParserImpl;
struct JsonParserRefImpl;
class JsonParserRef;


class JsonParser
{
    std::unique_ptr<JsonParserImpl> pImpl;

public:
    JsonParser(const char* jsonFilePath);
    ~JsonParser();
    
    JsonParserRef operator[](uint32_t index) const;
    JsonParserRef operator[](std::string_view key) const;
};


class JsonParserRef
{
    friend class JsonParser;
    std::unique_ptr<JsonParserRefImpl> pImpl;

public:
    JsonParserRef(std::unique_ptr<JsonParserRefImpl> impl);
    ~JsonParserRef();

    JsonParserRef operator[](uint32_t index) const;
    JsonParserRef operator[](std::string_view key) const;

    std::vector<float> parseNDArray(std::vector<uint32_t>& outShape) const;
    std::vector<float> parseNDArray() const;
};


#endif // JSONPARSER_H