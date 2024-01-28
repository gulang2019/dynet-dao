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

Expression compute_loss_lookup_as_param(
    LookupParameter& lp,
    const unsigned hdim,
    std::shared_ptr<ComputationGraph> cg 
) {
    unsigned batch_size = 2;
    std::vector<float> x(hdim*batch_size);
    std::vector<unsigned> indices(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        indices[i] = i;
    }
    for (int i = 0; i < hdim*batch_size; ++i) {
        x[i] = i;
    }
    Expression w = parameter(*cg, lp);
    w = transpose(w);
    // std::cout << "hdim:" << hdim << std::endl;
    // std::cout << "batch_size:" << batch_size << std::endl;
    // std::cout << "w:" << w.dim() << std::endl;
    Dim dim({hdim}, batch_size);
    Expression h = input(*cg, dim, x);
    // std::cout << "h:" << h.dim() << " bs: " << h.dim().bd << std::endl;
    Expression o = w * h;
    // std::cout << "o:" << o.dim() << std::endl;

    Expression loss_expr = sum_batches(pickneglogsoftmax(o, indices));
    return loss_expr;
}

Expression compute_loss(
    const unsigned ctx,
    std::shared_ptr<ComputationGraph> cg,
    MLP& mlp, 
    LookupParameter& lp) {
    std::vector<unsigned> x(ctx), y(ctx);
    for (int i = 0; i < ctx; ++i) {
        x[i] = i;
        y[i] = x[i];
    }
    Expression h = lookup(*cg, lp, x);
    h = mlp(cg, h);
    Expression lp_o = parameter(*cg, lp);
    lp_o = transpose(lp_o);
    Expression o = lp_o * h;
    Expression loss_expr = (sum_batches(pickneglogsoftmax(o, y)));
    return loss_expr;
}

void ground_truth_seq2seq() {
    unsigned ctx = 2;
    unsigned hdim = 2;
    unsigned n_vocab = 3;
    ParameterCollection params;
    AdamTrainer trainer(params);
    LookupParameter lp = params.add_lookup_parameters(n_vocab, {hdim});
    MLP mlp(hdim, hdim, params);

    for (int i = 0 ; i < 50; i++){
        std::shared_ptr<ComputationGraph> cg = std::make_shared<ComputationGraph>();
        auto loss_expr = compute_loss(ctx, cg, mlp, lp);
        // auto loss_expr = compute_loss_lookup_as_param(lp, hdim, cg);
        auto loss = cg->forward(loss_expr);
        cg->backward(loss_expr);
        trainer.update();
        std::cout << "loss " << i << ": " << as_vector(loss)[0] << std::endl;
    }
}

void test_seq2seq() {
    unsigned ctx = 2;
    unsigned hdim = 2;
    unsigned n_vocab = 3;
    ParameterCollection params;
    AdamTrainer trainer(params);
    LookupParameter lp = params.add_lookup_parameters(n_vocab, {hdim});
    MLP mlp(hdim, hdim, params);
    Engine engine(&trainer);

    for (int i = 0 ; i < 50; i++){
        std::shared_ptr<ComputationGraph> cg = std::make_shared<ComputationGraph>();
        auto loss_expr = compute_loss(ctx, cg, mlp, lp);
        // auto loss_expr = compute_loss_lookup_as_param(lp, hdim, cg);
        auto& loss = engine.symbolic_forward(cg, loss_expr);
        engine.symbolic_backward(cg, loss_expr);
        engine.symbolic_update();
        engine.run();
        std::cout << "loss " << i << ": " << engine.as_vector(loss)[0] << std::endl;
    }
    engine.report();
}


/*example usage: ./dao/tests/test_seq2seq --dynet-seed 1 --dao-verbose 0*/

int main(int argc, char** argv) {
    dynet::initialize(argc, argv);
    // printf("ground_truth\n");
    // ground_truth_sgd();
    if (DAO::use_dao) {
        printf("dao\n");
        test_seq2seq();
    }
    else {
        printf("sgd\n");
        ground_truth_seq2seq();
    }
    // ground_truth_seq2seq();
    // printf("sgd\n");
    return 0;
}