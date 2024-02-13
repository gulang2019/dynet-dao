#include <DAO/DAO.h> 
#include <DAO/engine.h>
#include <DAO/utils.h>

#include <dynet/dynet.h>
#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/devices.h>
#include <dynet/param-nodes.h>

#include <vector>
#include <cuda_runtime.h>

using namespace DAO;
using namespace dynet;
using std::vector;  

struct MLP {
    Parameter p_W;
    Parameter p_b;

    MLP(int input_dim, int hidden_dim, ParameterCollection& params) {
        p_W = params.add_parameters({hidden_dim, input_dim});
        p_b = params.add_parameters({hidden_dim});
    }

    Expression operator()(std::shared_ptr<dynet::ComputationGraph>cg, const Expression& x) const {
        Expression W = parameter(*cg, p_W);
        Expression b = parameter(*cg, p_b);
        Expression h = tanh(W*x + b);
        return h;
    }
};

void check_all_same(
    const std::vector<float>& value,
    const std::vector<float>& ground_truth 
) {
    std::cout << "value: ";
    int i = 0;
    for (auto v : value) {
        if (i > 5) break;
        i++;
        std::cout << v << " ";
    }
    std::cout << std::endl;
    std::cout << "ground truth: ";
    i = 0;
    for (auto v : ground_truth) {
        if (i > 5) break;
        i++;
        std::cout << v << " ";
    }
    std::cout << std::endl;
    // DAO_ASSERT(value.size() == ground_truth.size(), "value.size() != ground_truth.size()");
    // for (int i = 0; i < value.size(); ++i) {
    //     DAO_ASSERT(abs(value[i] - ground_truth[i]) < 1e-3, "%f != %f", value[i], ground_truth[i]);
    // }
}

void check(
    DAO::Engine& engine, 
    std::shared_ptr<ComputationGraph> cg, 
    dynet::Expression& expr) {
    int i = 0;
    for (auto& node : cg->nodes) {
        DAO_INFO("N%d: %s", i++, node->as_dummy_string().c_str());
    }
    const Tensor& ground_truth = cg->forward(expr);
    auto ground_truth_value = as_vector(ground_truth);
    const Tensor& tensor = engine.symbolic_forward(cg, expr);
    engine.run();
    DAO_INFO("tensor: %p %u %s", tensor.v, tensor.d.size(), tensor.device->name.c_str());
    auto value = engine.as_vector(tensor);
    check_all_same(value, ground_truth_value);
}

void check_fwd_bwd(
    DAO::Engine& engine, 
    std::shared_ptr<ComputationGraph> cg, 
    dynet::Expression& expr
) {
    Timer timer; 
    int i = 0;
    for (auto& node : cg->nodes) {
        DAO_INFO("N%d: %s", i++, node->as_dummy_string().c_str());
    }
    timer.start("DyNet");
    const Tensor& ground_truth_fwd = cg->forward(expr);
    cudaDeviceSynchronize();
    cg->backward(expr);
    
    timer.stop("DyNet");
    std::vector<std::vector<float> > ground_truth;
    std::vector<std::vector<float> > value;
    for (dynet::VariableIndex idx = 0; idx < cg->nodes.size(); ++idx) {
        auto& grad_tensor = cg->get_gradient(idx);
        auto grad_value = as_vector(grad_tensor);
        int i = 0;
        std::stringstream ss;
        ss << "grad " << idx << ": ";
        for (auto v : grad_value) {
            if (i++ > 5) break;
            ss << v << " ";
        }
        DAO_INFO("%s", ss.str().c_str());
    }
    for (auto idx: cg->parameter_nodes) {
        
        auto& node = cg->nodes[idx];
        ParameterNode* pnode = static_cast<ParameterNode*>(node);
        if (pnode->params.p) {
            pnode->params.get_storage().g.v = (float*)get_allocator()->prepare(&pnode->params.get_storage().g, DAO::TensorRecord::PARAMETER);
            ground_truth.push_back(as_vector(pnode->params.get_storage().g));
            pnode->params.get_storage().clear();
        }
        else { 
            pnode->lparams.get_storage().all_grads.v = (float*)get_allocator()->prepare(&pnode->lparams.get_storage().all_grads, DAO::TensorRecord::OPTIMIZER_STATE);
            ground_truth.push_back(as_vector(pnode->lparams.get_storage().all_grads));
            pnode->lparams.get_storage().clear();
        }
    }

    const Tensor& value_fwd = engine.symbolic_forward(cg, expr);
    engine.symbolic_backward(cg, expr);

    timer.start("DAO");
    engine.run();
    cudaDeviceSynchronize();
    timer.stop("DAO");
    

    check_all_same(engine.as_vector(value_fwd), as_vector(ground_truth_fwd));
    
    for (auto idx: cg->parameter_nodes) {
        auto& node = cg->nodes[idx];
        ParameterNode* pnode = static_cast<ParameterNode*>(node);
        if (pnode->params.p) {
            pnode->params.get_storage().g.v = (float*)get_allocator()->prepare(&pnode->params.get_storage().g, DAO::TensorRecord::PARAMETER);
            value.push_back(engine.as_vector(pnode->params.get_storage().g));
            pnode->params.get_storage().clear();
        }
        else { 
            pnode->lparams.get_storage().all_grads.v = (float*)get_allocator()->prepare(&pnode->lparams.get_storage().all_grads, DAO::TensorRecord::OPTIMIZER_STATE);
            value.push_back(engine.as_vector(pnode->lparams.get_storage().all_grads));
            pnode->lparams.get_storage().clear();
        }
    }

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        DAO_INFO("Check %s %d", cg->nodes[cg->parameter_nodes[i]]->as_dummy_string().c_str(), i);
        check_all_same(value[i], ground_truth[i]);
    }

    engine.report();
    timer.show();
}

void test_mlp_fwd() {
    unsigned I = 1024, O = 1;

    ParameterCollection params; 
    Parameter p_W = params.add_parameters({O, I});

    Engine engine;
    std::vector<MLP> layers;
    for (int i = I; i != O; i /= 2) {
        layers.emplace_back(i, i/2, params);
    }
    std::shared_ptr<ComputationGraph> cg = std::make_unique<ComputationGraph>();
    vector<dynet::real> x_values(I);
    for (int i = 0; i < I; ++i) {
        x_values[i] = i;
    }
    Expression x = input(*cg, {I}, &x_values);
    for (auto& layer : layers) {
        x = layer(cg, x);
    }
    check(engine, cg, x);
}

void test_mlp_bwd() {
    unsigned I = 1024, O = 1;

    ParameterCollection params; 
    Parameter p_W = params.add_parameters({O, I});

    Engine engine;
    std::vector<MLP> layers;
    // for(int i = 0; i < 100; ++i) {
    //     layers.emplace_back(1, 1, params);
    // }
    for (int i = I; i != O; i /= 2) {
        layers.emplace_back(i, i/2, params);
    }
    // layers.emplace_back(I,O, params);
    // layers.emplace_back(I/2, O, params);
    std::shared_ptr<ComputationGraph> cg = std::make_unique<ComputationGraph>();
    vector<dynet::real> x_values(I);
    for (int i = 0; i < I; ++i) {
        x_values[i] = i + 1;
    }
    Expression x = input(*cg, {I}, &x_values);
    for (auto& layer : layers) {
        x = layer(cg, x);
    }

    const Tensor& value_fwd = engine.symbolic_forward(cg, x);
    engine.symbolic_backward(cg, x);

    engine.run();
    engine.report();
    // check_fwd_bwd(engine, cg, x);
}

// void test_xor() {
//     unsigned HIDDEN_SIZE = 8;
//     unsigned ITERATIONS = 100;

//     ParameterCollection params; 
//     SimpleSGDTrainer trainer(params); 
//     Parameter p_W = params.add_parameters({HIDDEN_SIZE, 2});
//     Parameter p_b = params.add_parameters({HIDDEN_SIZE});
//     Parameter p_V = params.add_parameters({1, HIDDEN_SIZE});
//     Parameter p_a = params.add_parameters({1});
//     vector<real> losses;

//     Engine engine;
//     std::shared_ptr<ComputationGraph> cg = std::make_unique<ComputationGraph>();
//     Expression W = parameter(*cg, p_W);
//     Expression b = parameter(*cg, p_b);
//     Expression V = parameter(*cg, p_V);
//     Expression a = parameter(*cg, p_a);
//     vector<dynet::real>* x_values = new vector<dynet::real>(2);
//     Expression x = input(*cg, {2}, x_values);
//     dynet::real* y_value = new dynet::real;  // Set y_value to change the target output.
//     Expression y = input(*cg, y_value);

//     Expression h = tanh(W*x + b);
//     Expression y_pred = V*h + a;
//     Expression loss_expr = squared_distance(y_pred, y);

//     unsigned iter = 0;
//     bool x1 = iter % 2;
//     bool x2 = (iter / 2) % 2;
//     (*x_values)[0] = x1 ? 1 : -1;
//     (*x_values)[1] = x2 ? 1 : -1;
//     *y_value = (x1 != x2) ? 1 : -1;

//     engine.symbolic_forward(cg, loss_expr);
//     const Tensor& loss = engine.run();
//     const Tensor& ground_truth = cg->forward(loss_expr);
//     auto loss_value = as_scalar(loss); 
//     auto ground_truth_value = as_scalar(ground_truth);
//     std::cout << "loss: " << loss_value << std::endl;
//     std::cout << "ground_truth: " << ground_truth_value << std::endl;
// }
/**
a @ W + b
<=>
multiply(a, W).then(add(b))
1. when the tensor deconstructs, we release the tensor
2. when the tensor does not have future memory access, we realease the tensor; 
    In this case, we need to constrain the user's access to the tensors; 
    DAO may release this tensor because it infer the tensor's access pattern due to the network training; 
    For intermidiate tensor 
        1. you must call tocpu() immediately after the forward() call; 
        2. we only guarantee the tensor after 
    DAO::tocpu(tensor) is the only way to access the tensor; 
*/
int main(int argc, char** argv) {
    
    dynet::initialize(argc, argv);
    test_mlp_bwd();
}