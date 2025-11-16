#include "jsonParser.h"
#include <nlohmann/json.hpp>
#include <fstream>



struct JsonParserImpl 
{
    nlohmann::json jsonData;

    JsonParserImpl(const char* jsonFilePath) 
    {
        std::ifstream file(jsonFilePath);
        if (!file.is_open())
            throw std::runtime_error("Could not open JSON file: " + std::string(jsonFilePath));
        file >> jsonData;
    }
};

struct JsonParserRefImpl 
{
    nlohmann::json& jsonData;
    JsonParserRefImpl(nlohmann::json& j) : jsonData(j) {}
};



JsonParser::~JsonParser() = default;
JsonParser::JsonParser(const char* jsonFilePath)
    : pImpl(std::make_unique<JsonParserImpl>(jsonFilePath)) 
{
}


JsonParserRef::~JsonParserRef() = default;
JsonParserRef::JsonParserRef(std::unique_ptr<JsonParserRefImpl> impl)
    : pImpl(std::move(impl)) 
{
}


JsonParserRef JsonParser::operator[](uint32_t index) const
{
    if (!pImpl->jsonData.is_array())
        throw std::runtime_error("JSON key is not an array: " + std::string(pImpl->jsonData.dump()));

    if (index >= pImpl->jsonData.size())
        throw std::out_of_range("Index out of range for JSON array");

    return JsonParserRef(std::make_unique<JsonParserRefImpl>(pImpl->jsonData[index]));
}


JsonParserRef JsonParser::operator[](std::string_view key) const
{
    if (!pImpl->jsonData.is_object())
        throw std::runtime_error("JSON key is not an object: " + std::string(pImpl->jsonData.dump()));

    auto it = pImpl->jsonData.find(key);
    if (it == pImpl->jsonData.end())
        throw std::out_of_range("Key not found in JSON object: " + std::string(key));

    return JsonParserRef(std::make_unique<JsonParserRefImpl>(*it));
}


JsonParserRef JsonParserRef::operator[](uint32_t index) const
{
    if (!pImpl->jsonData.is_array())
        throw std::runtime_error("JSON key is not an array: " + std::string(pImpl->jsonData.dump()));

    if (index >= pImpl->jsonData.size())
        throw std::out_of_range("Index out of range for JSON array");

    return JsonParserRef(std::make_unique<JsonParserRefImpl>(pImpl->jsonData[index]));
}


JsonParserRef JsonParserRef::operator[](std::string_view key) const
{
    if (!pImpl->jsonData.is_object())
        throw std::runtime_error("JSON key is not an object: " + std::string(pImpl->jsonData.dump()));

    auto it = pImpl->jsonData.find(key);
    if (it == pImpl->jsonData.end())
        throw std::out_of_range("Key not found in JSON object: " + std::string(key));

    return JsonParserRef(std::make_unique<JsonParserRefImpl>(*it));
}



std::vector<float> JsonParserRef::parseNDArray(std::vector<uint32_t>& outShape) const
{
    if (!pImpl->jsonData.is_array())
        throw std::runtime_error("JSON key is not an array: " + std::string(pImpl->jsonData.dump()));

    auto* current = &pImpl->jsonData;
    for (; current->is_array() && !current->empty(); current = &(*current)[0]) 
    {
        outShape.push_back(static_cast<uint32_t>(current->size()));
    }

    if (!current->is_number_float()) 
        throw std::runtime_error("JSON array does not contain float values");

    size_t totalSize = 1;
    for (uint32_t dim: outShape) 
        totalSize *= dim;

    std::vector<float> data;
    data.reserve(totalSize);

    auto flatten = [&](auto& self, const nlohmann::json& node) -> void
    {
        if (node.is_array()) 
            for (const auto& child : node)
                self(self, child);
        else if (node.is_number_float() || node.is_number_integer()) 
            data.push_back(node.get<float>());
        else 
            throw std::runtime_error("Encountered non-numeric value in JSON array");
    };
    flatten(flatten, pImpl->jsonData);

    return data;
}


std::vector<float> JsonParserRef::parseNDArray() const
{
    std::vector<uint32_t> shape;
    return parseNDArray(shape);
}

