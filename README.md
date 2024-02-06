## DyNet-DAO 

This repo implements dynamic activation offloading on-top-of DyNet. 

### Build

Preparation
```
# load CUDA 11.1;
cd ~ 
source cuda.sh 11.1 
```

Install Eigen 
```
mkdir eigen
cd eigen
wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
unzip eigen-b2e267dc99d4.zip
```

Build 
```
git clone https://github.com/clab/dynet.git
cd dynet
mkdir build
cd build
# Run CMake
# -DENABLE_BOOST=ON in combination with -DENABLE_CPP_EXAMPLES=ON also
# compiles the multiprocessing c++ examples
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DENABLE_BOOST=ON -DENABLE_CPP_EXAMPLES=ON -DBACKEND=cuda -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_DAO=ON
# Compile using 2 processes
make -j 2
# Test with an example
./examples/xor
```

Run mnist 
```
# prepare datasets 
ln -s /ssd1/siyuanch/workspace/dynet-dao/datasets datasets 
cd build/examples 
./mnist -t ../../datasets/mnist/train-images.idx3-ubyte -d ../../datasets/mnist/t10k-images.idx3-ubyte -tl ../../datasets/mnist/train-labels.idx1-ubyte -dl ../../datasets/mnist/t10k-labels.idx1-ubyte --batch_size 128 -N 20
```

Transformer Example 
```
# cd <repo dir>
./build/examples/transformer-train -c models/iwslt-envi/config.txt --parameters models/iwslt-envi/en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml150_ffrelu &>models/iwslt-envi/log.en-vi.transformer.h2_l2_u128_do010101010001_att1_ls00_pe1_ml150_ffrelu
```

Example: fine-tune gpt2 with lora and skip rate 0.2
```bash
skip_r=0.2
# cd <repo dir>
mkdir -p models/gpt2-124M
cp /home/siyuanch/ssd/workspace_zelongg/dynet-dao/models/gpt2-124M/hparams.ini models/gpt2-124M
# TODO: modify hparams.ini for epochs, bs and log frequency
mkdir -p models/gpt2-124M-$skip_r  # prepare initial checkpoint
echo "768 12 12 4 0 0.1 $skip_r 0 0 0.1 1 1024 1 1 0 models/gpt2-124M-$skip_r/model.params" > models/gpt2-124M-$skip_r/model.config
cp /ssd1/siyuanch/workspace_zelongg/DAO/models/124M/dynet-model.params models/gpt2-124M-$skip_r/model.params
# Add --train-percent 10 to below cmd for faster run
./build/examples/transformer-lm -c models/gpt2-124M/hparams.ini --model-path models/gpt2-124M-$skip_r --attn-lora-r 2 --attention-dropout-p $skip_r --ff-dropout-p 0 --reset-if-stuck --use-smaller-minibatch 2>&1 | tee models/gpt2-124M-$skip_r/train.log

# Run transformer with dao 
./build/examples/transformer-lm --train-percent 3 --use_offload --dao-gpu-mem 16384  --dao-verbose 0 -c models/gpt2-124M/hparams.ini --attn-lora-r 2 --attention-dropout-p 0.2 --ff-dropout-p 0.2 --reset-if-stuck --use-smaller-minibatch --dynet-seed 1 2>&1 | tee models/gpt2-124M-0.2/train.log
```

#### DAO command line arguments
- `--use_offload`: enable DAO's offloading backend; otherwise, we fallback to dynet's backend;
- `--dao-gpu-mem [int]`: the gpu memory size for DAO's backend in MB;
- `--dao-cpu-mem [int]`: the cpu memory size for DAO's backend in MB;
- `--dao-verbose [int=0]`: the verbose level of DAO, default 0;
- `--dao-debug`: enable a lot of assertions of DAO;
- `--dynet-seed [int=0]`: the random seed; default = 0, meaning random;  
- `--dao-profile 1`: enable tracing of kernels; In your c++ application code, #include <DAO/DAO.h> and use DAO::profiler.dump(std::string name) method to dump the traces into a "name.traces" file.

## GPT2 scripts;
We have a script to generate script for runnning gpt2; use `python examples/gpt2/gen_script.py --help` to look at the usage. 
```bash
python examples/gpt2/gen_script.py --name gpt2-124M -c models/gpt2-124M/hparams.ini --gpu-mem 3 --attn-lora-r 4 --attention-dropout-p 0.0 0.4 0.8 --ff-dropout-p 0.0 0.4 0.8 --update-freq 8 --bs 2048 --script-name run_linear
```

## DAO API
We use `Engine` to train the model by delaying the forward/backward/update. An usage The API can be seen in the [header](dao/include/engine.h), an example usage can be seen at [here](dao/tests/test_seq2seq.cc).

Also, we add a feature to dynet to set if a parameter is trainable. `dynet::ParameterCollection::set_default_updated(bool trainable)` is used to set if the parameters is by default trainable or not. To specify if a parameter is trainable or not, one can use this api to add parameters into the collection:
```c++
/**
   * \brief Add parameters with custom initializer
   *
   * \param d Shape of the parameter
   * \param init Custom initializer
   * \param name Name of the parameter
   * \param device Device placement for the parameter
   * \param trainable Whether the parameter is trainable or not
   *
   * \return Parameter object to be used in the computation graph
   */
  Parameter ParameterCollection::add_parameters(const Dim& d, const ParameterInit & init,
                           const std::string & name, Device *device, bool trainable);
```


<!-- <div align="center">
  <img alt="DyNet" src="doc/source/images/dynet_logo.png"><br><br>
</div>

---

[![Build Status (Travis CI)](https://travis-ci.org/clab/dynet.svg?branch=master)](https://travis-ci.org/clab/dynet)
[![Build Status (AppVeyor)](https://ci.appveyor.com/api/projects/status/github/clab/dynet?svg=true)](https://ci.appveyor.com/project/danielh/dynet-c3iuq)
[![Build Status (Docs)](https://readthedocs.org/projects/dynet/badge/?version=latest)](http://dynet.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/dyNET.svg)](https://badge.fury.io/py/dyNET)

The Dynamic Neural Network Toolkit

- [General](#general)
- [Installation](#installation)
  - [C++](#c-installation)
  - [Python](#python-installation)
- [Getting Started](#getting-started)
- [Citing](#citing)
- [Releases and Contributing](#releases-and-contributing)


## General

DyNet is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python) and is designed to be efficient when run on either CPU or GPU, and to work well with networks that have dynamic structures that change for every training instance. For example, these kinds of networks are particularly important in natural language processing tasks, and DyNet has been used to build state-of-the-art systems for [syntactic parsing](https://github.com/clab/lstm-parser), [machine translation](https://github.com/neubig/lamtram), [morphological inflection](https://github.com/mfaruqui/morph-trans), and many other application areas.

Read the [documentation](http://dynet.readthedocs.io/en/latest/) to get started, and feel free to contact the [dynet-users group](https://groups.google.com/forum/#!forum/dynet-users) group with any questions (if you want to receive email make sure to select "all email" when you sign up). We greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull request through the [github page](http://github.com/clab/dynet).

You can also read more technical details in our [technical report](https://arxiv.org/abs/1701.03980).

## Getting started

You can find tutorials about using DyNet [here (C++)](http://dynet.readthedocs.io/en/latest/tutorial.html#c-tutorial) and [here (python)](http://dynet.readthedocs.io/en/latest/tutorial.html#python-tutorial), and [here (EMNLP 2016 tutorial)](https://github.com/clab/dynet_tutorial_examples).

One aspect that sets DyNet apart from other tookits is the **auto-batching** feature. See the [documentation](http://dynet.readthedocs.io/en/latest/minibatch.html) about batching.

The `example` folder contains a variety of examples in C++ and python.


## Installation

DyNet relies on a number of external programs/libraries including CMake and
Eigen. CMake can be installed from standard repositories.

For example on **Ubuntu Linux**:

    sudo apt-get install build-essential cmake

Or on **macOS**, first make sure the Apple Command Line Tools are installed, then
get CMake, and Mercurial with either homebrew or macports:

    xcode-select --install
    brew install cmake  # Using homebrew.
    sudo port install cmake # Using macports.

On **Windows**, see [documentation](http://dynet.readthedocs.io/en/latest/install.html#windows-support).

To compile DyNet you also need a [specific version of the Eigen
library](https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip). **If you use any of the
released versions, you may get assertion failures or compile errors.**
You can get it easily using the following command:

    mkdir eigen
    cd eigen
    wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
    unzip eigen-b2e267dc99d4.zip


### C++ installation

You can install dynet for C++ with the following commands

    # Clone the github repository
    git clone https://github.com/clab/dynet.git
    cd dynet
    mkdir build
    cd build
    # Run CMake
    # -DENABLE_BOOST=ON in combination with -DENABLE_CPP_EXAMPLES=ON also
    # compiles the multiprocessing c++ examples
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DENABLE_CPP_EXAMPLES=ON
    # Compile using 2 processes
    make -j 2
    # Test with an example
    ./examples/xor

For more details refer to the [documentation](http://dynet.readthedocs.io/en/latest/install.html#building)

### Python installation

You can install DyNet for python by using the following command

    pip install git+https://github.com/clab/dynet#egg=dynet

For more details refer to the [documentation](http://dynet.readthedocs.io/en/latest/python.html#installing-dynet-for-python)

## Citing

If you use DyNet for research, please cite this report as follows:

    @article{dynet,
      title={DyNet: The Dynamic Neural Network Toolkit},
      author={Graham Neubig and Chris Dyer and Yoav Goldberg and Austin Matthews and Waleed Ammar and Antonios Anastasopoulos and Miguel Ballesteros and David Chiang and Daniel Clothiaux and Trevor Cohn and Kevin Duh and Manaal Faruqui and Cynthia Gan and Dan Garrette and Yangfeng Ji and Lingpeng Kong and Adhiguna Kuncoro and Gaurav Kumar and Chaitanya Malaviya and Paul Michel and Yusuke Oda and Matthew Richardson and Naomi Saphra and Swabha Swayamdipta and Pengcheng Yin},
      journal={arXiv preprint arXiv:1701.03980},
      year={2017}
    }


## Contributing

We welcome any contribution to DyNet! You can find the contributing guidelines [here](http://dynet.readthedocs.io/en/latest/contributing.html) -->
