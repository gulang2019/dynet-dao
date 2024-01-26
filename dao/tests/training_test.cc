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

    MLP(unsigned input_dim, unsigned hidden_dim, ParameterCollection& params) {
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

void ground_truth_sgd() {
    unsigned I = 1024, O = 1;
    ParameterCollection params;
    AdamTrainer trainer(params);
    Engine engine(&trainer);
    
    std::vector<MLP> layers;
    for (int i = I; i != O; i /= 2) {
        layers.emplace_back(i, i/2, params);
    }

    for (int i = 0 ; i < 50; i++){
        std::vector<float> x(I);
        float y = 1;
        for (int j = 0; j < I; ++j) {
            x[j] = rand() % 2 - 1;
            y = y*x[j];
        }
        std::shared_ptr<ComputationGraph> cg = std::make_shared<ComputationGraph>();
        Expression x_expr = input(*cg, {I}, x);
        for (auto& layer : layers) {
            x_expr = layer(cg, x_expr);
        }
        Expression y_expr = input(*cg, y);
        Expression loss_expr = squared_distance(x_expr, y_expr);
        auto& loss = cg->forward(loss_expr);
        cg->backward(loss_expr);
        // if (i % 2 == 0){
            trainer.update();
            std::cout << "loss: " << as_scalar(loss) << std::endl;
        // }
    }
}

void test_sgd() {
    int I = 1024, O = 1;
    ParameterCollection params;
    AdamTrainer trainer(params);
    std::vector<MLP> layers;
    for (int i = I; i != O; i /= 2) {
        layers.emplace_back((unsigned)i, (unsigned)i>>1, params);
    }

    Engine engine(&trainer);

    for (int i = 0 ; i < 50; i++){
        std::vector<float> x(I);
        float y = 1;
        for (int j = 0; j < I; ++j) {
            x[j] = rand() % 2 - 1;
            y = y*x[j];
        }
        std::shared_ptr<ComputationGraph> cg = std::make_shared<ComputationGraph>();
        Expression x_expr = input(*cg, {I}, x);
        for (auto& layer : layers) {
            x_expr = layer(cg, x_expr);
        }
        Expression y_expr = input(*cg, y);
        Expression loss_expr = squared_distance(x_expr, y_expr);
        auto& loss = engine.symbolic_forward(cg, loss_expr);
        engine.symbolic_backward(cg, loss_expr);
        engine.symbolic_update();
        engine.run();
        std::cout << "loss: " << engine.as_vector(loss)[0] << std::endl;
    }
    engine.report();
}

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    // printf("ground_truth\n");
    // ground_truth_sgd();
    printf("sgd\n");
    test_sgd();
    return 0;
}