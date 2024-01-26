#include <DAO/DAO.h> 

#include <dynet/dynet.h>
#include <dynet/training.h>
#include <dynet/expr.h>

#include <vector>

using namespace DAO;
using namespace dynet;
using std::vector;  

void test_add() {
    unsigned HIDDEN_SIZE = 8;
    unsigned ITERATIONS = 100;

    ParameterCollection params; 
    SimpleSGDTrainer trainer(params); 
    Parameter p_W = params.add_parameters({HIDDEN_SIZE, 2});
    Parameter p_b = params.add_parameters({HIDDEN_SIZE});
    Parameter p_V = params.add_parameters({1, HIDDEN_SIZE});
    Parameter p_a = params.add_parameters({1});
    vector<real> losses;
    for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
        std::shared_ptr<ComputationGraph> cg = std::make_unique<ComputationGraph>();
        Expression W = parameter(*cg, p_W);
        Expression b = parameter(*cg, p_b);
        Expression V = parameter(*cg, p_V);
        Expression a = parameter(*cg, p_a);
        vector<dynet::real>* x_values = new vector<dynet::real>(2);
        Expression x = input(*cg, {2}, x_values);
        dynet::real* y_value = new dynet::real;  // Set y_value to change the target output.
        Expression y = input(*cg, y_value);

        Expression h = tanh(W*x + b);
        Expression y_pred = V*h + a;
        Expression loss_expr = squared_distance(y_pred, y);

        bool x1 = iter % 2;
        bool x2 = (iter / 2) % 2;
        (*x_values)[0] = x1 ? 1 : -1;
        (*x_values)[1] = x2 ? 1 : -1;
        *y_value = (x1 != x2) ? 1 : -1;
        // construct graph 
        const Tensor& loss = cg->forward(loss_expr);        
        if (iter % 10 == 0) {
            losses.push_back(as_scalar(loss));
            std::cout << "loss: " << losses.back() << std::endl;
        }
        cg->backward(loss_expr);
        // DAO::sync();
        trainer.update();
        /**
         engine.symbolic_forward(cg, loss);
         engine.symbolic_backward();
         engine.symbolic_update();
         engine.forward();
        */
    }
    
    // std::shared_ptr<ComputationGraph> cg = std::make_unique<ComputationGraph>();
    // Expression W = parameter(*cg, p_W);
    // Expression b = parameter(*cg, p_b);
    // Expression V = parameter(*cg, p_V);
    // Expression a = parameter(*cg, p_a);
    // vector<dynet::real> x_values(2);
    // Expression x = input(*cg, {2}, &x_values);
    // dynet::real y_value;  // Set y_value to change the target output.
    // Expression y = input(*cg, &y_value);

    // Expression h = tanh(W*x + b);
    // Expression y_pred = V*h + a;
    // for (int i = 0; i < 4; i++) {
    //     bool x1 = i % 2;
    //     bool x2 = (i / 2) % 2;
    //     x_values[0] = x1 ? 1 : -1;
    //     x_values[1] = x2 ? 1 : -1;
    //     y_value = (x1 != x2) ? 1 : -1;
    //     const Tensor& pred = cg->forward(y_pred);
    //     DAO::sync();
    //     std::cout << x1 << ", " << x2 << ", pred: " << as_scalar(pred) << std::endl; 
    // }
}
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
    test_add();
}