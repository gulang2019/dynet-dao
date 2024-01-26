#include <DAO/generator.h>
#include <DAO/utils.h>
#include <DAO/globals.h>

namespace DAO {

void push(std::vector<dynet::Tensor*>& vec, const std::vector<const dynet::Tensor*>& to_push) {
    for (const auto& tensor : to_push) {
        // Assuming you want to add the const pointers to the non-const vector
        // You need to ensure this is safe and does not lead to undefined behavior.
        vec.push_back(const_cast<dynet::Tensor*>(tensor));
    }
}

void push(std::vector<TensorUID>& vec, const std::vector<TensorUID>& to_push) {
    vec.insert(vec.end(), to_push.begin(), to_push.end());
}

void push(std::vector<TensorUID>& vec, TensorUID to_push) {
    vec.push_back(to_push);
}


} // DAO 