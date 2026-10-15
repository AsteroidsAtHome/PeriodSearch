
#include <regex>
#include <sstream>
#include <string>

std::string transformKernelCode(const std::string &kernel_code)
{
    // Regular expressions to match tabs and multiple spaces
    std::regex tabRegex("\\t");
    std::regex spacesRegex(" {2,}");
    //std::regex quoteRegex("\"");

    // Replace tabs with a single space
    std::string transformed_code = std::regex_replace(kernel_code, tabRegex, " ");
    // Replace multiple spaces with a single space
    transformed_code = std::regex_replace(transformed_code, spacesRegex, " ");
    // Escape double quotes
    //transformed_code = std::regex_replace(transformed_code, quoteRegex, "\\\"");

    return transformed_code;
}

//std::string transformKernelCode(const std::string &kernel_code)
//{
//    // Regular expressions to match tabs and multiple spaces
//    std::regex tabRegex("\\t");
//    std::regex spacesRegex(" {2,}");
//    std::regex quoteRegex("\"");
//
//    // Replace tabs with a single space
//    std::string transformed_code = std::regex_replace(kernel_code, tabRegex, " ");
//    // Replace multiple spaces with a single space
//    transformed_code = std::regex_replace(transformed_code, spacesRegex, " ");
//    // Escape double quotes
//    transformed_code = std::regex_replace(transformed_code, quoteRegex, "\\\"");
//
//    // Resulting string to hold the transformed code
//    std::string result;
//
//    // Split the transformed_code into lines and append each line with formatting
//    std::istringstream stream(transformed_code);
//    std::string line;
//    while (std::getline(stream, line))
//    {
//        result += "\"" + line + "\\n\"\n";
//    }
//
//    // End the variable declaration properly
//    //result += ";";
//
//    return result;
//}
