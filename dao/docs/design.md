```
ParameterCollection params;
AdamTrainer trainer(params);
Model model; 

for (i = 0; i < num_epochs; ++i)
    for (si = 0; si < num_batches; ++si) {
        input, label = data_loader.get(si);
        ComputationGraph cg;
        Expression input = input(cg, {shape}, input);
        Expression loss = model.loss(input, label);
        loss = cg.forward();

        cg.backward();
        trainer.update();
    }
```

|procedure | optimizer states | intermidiate tensors | parameter | gradients |
|- | - | - | - | - | 
|ParameterCollection.add_parameters | | | Allocation | Allocation | 
|populate| | |WRITE||
|Trainer(model)| Allocation | | | | 
|cg = Graph Definition() | 
|cg.forward()| | Allocation, R/W | READ || 
|cg.backward() | | Allocation, R/W | READ | WRITE |
|update() | R/W | | WRITE | R/W | 

Allocation of different tensors are done by us.
Before visiting every tensor, we need to prepare it.
We need to give the  

```
ParameterCollection.add_parameters
device->allocate_tensor(DeviceMempool::PS, values);
tens.v = (float*)pools[(int)mp]->allocate(tens.d.size() * sizeof(float));

populate
TensorTools::set_elements
cudaMemcpyAsync(v.v, &vec[0], sizeof(real) * vec.size(), cudaMemcpyHostToDevice);
TensorTools::zero 
void TensorTools::constant_dev(const MyDevice & dev, Tensor& d, float c) {
  tvec(d).device(*dev.edevice) = tvec(d).constant(c);
}

Trainer.update()
alloc_impl();
allocate_shadow_parameters()
p.device->allocate_tensor(DeviceMempool::PS, h);
alloc_lookup_impl();
TensorTools::zero() 

cg.forward() 
void* aux_mem = nullptr;
// Is the node requesting extra memory?
size_t aux_size = node->aux_storage_size();
if (aux_size) {
    aux_mem = node_fx_pools[(int)DeviceMempool::FXS]->allocate(aux_size);
    if (aux_mem == nullptr)
    DYNET_RUNTIME_ERR("Ran out of auxiliary memory when executing node "
                        << num_nodes_evaluated);
}
node->aux_mem = aux_mem;

// Compute f(xs) and store to node_fx.
node->forward(xs, node_fx);

cg.backward() 



```

```
ParameterCollection.add_parameters: 
    device->allocate_mem
```