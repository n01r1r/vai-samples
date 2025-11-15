#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <string>

using json = nlohmann::json;

// Helper function to load JSON test data
inline json loadTestData(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open test data file: " + filename);
    }

    json data;
    file >> data;
    return data;
}

// Helper function to convert JSON array to std::vector<float>
inline void flattenJson(const json& j, std::vector<float>& result)
{
    if (j.is_array()) {
        for (const auto& elem : j) {
            flattenJson(elem, result);
        }
    } else if (j.is_number()) {
        result.push_back(j.get<float>());
    }
}

inline std::vector<float> jsonToVector(const json& j)
{
    std::vector<float> result;
    flattenJson(j, result);
    return result;
}

#endif // TEST_HELPERS_H
