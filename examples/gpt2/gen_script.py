'''
./build_release/examples/transformer-lm --use_offload --dao-gpu-mem 1024 --dao-cpu-mem 4096 -c models/gpt2-124M/hparams.ini --model-path models/gpt2-124M-0.2 --attn-lora-r 2 --attention-dropout-p 0.2 --ff-dropout-p 0.2 --reset-if-stuck --use-smaller-minibatch --log-dir log g 2>&1
'''


import argparse
import os

example_usage = '''
example: python examples/gpt2/gen_script.py --name gpt2-124M -c models/gpt2-124M/hparams.ini --gpu-mem 4 --attn-lora-r 4 --attention-dropout-p 0.0 0.4 0.8 --ff-dropout-p 0.0 0.4 0.8 --update-freq 4 --bs 1024 --script-name run_stochastic_depth
'''

parser = argparse.ArgumentParser(description='Generate text from a trained GPT-2 model', epilog=example_usage)
parser.add_argument('--name', type=str, help='name of the model')
parser.add_argument('-c', '--config', type=str, help='path to the model config file')
parser.add_argument('--gpu-mem', type=int, nargs='+', default = [2,4,8,16],  help='GPU memory in GB')
parser.add_argument('--attn-lora-r', type=int, nargs = '+', default=[2,4,8], help='LoRA attention radius')
parser.add_argument('--attention-dropout-p', type=float, nargs='+', default=[0.0,0.4,0.8], help='attention dropout probability')
parser.add_argument('--ff-dropout-p', type=float, nargs='+', default=[0.0,0.4,0.8], help='feedforward dropout probability')
parser.add_argument('--update-freq', type=int, nargs='+', default=[4], help='update frequency')
parser.add_argument('--bs', type=int, nargs='+', default=[512,1024,2048], help='batch size')
parser.add_argument('--script-name', type=str, default='run', help='name of the script file')
parser.add_argument('--dropout-decay', action='store_true', help='use linear dropout decay')
parser.add_argument('--train-percent', default=100, type=int, help='train percent')
args = parser.parse_args()



cmds = []
cnt = 0
for bs in args.bs:
    for uf in args.update_freq:
        for gpu_mem in args.gpu_mem:
            for lora_r in args.attn_lora_r:
                for attn in args.attention_dropout_p:
                    for ff in args.ff_dropout_p:
                        cnt += 1
                        model_dir = f'models/{args.name}-{lora_r}-{attn}-{ff}-{uf}-{bs}-{gpu_mem}-{int(args.dropout_decay)}'
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        cmd = f"cp /ssd1/siyuanch/workspace_zelongg/DAO/models/124M/dynet-model.params {model_dir}\n"
                        cmd += f'./build_release/examples/transformer-lm -c {args.config} --use_offload --dao-gpu-mem {gpu_mem*1024} --dao-cpu-mem 16384 --minibatch-size {bs} --model-path {model_dir} --attn-lora-r {lora_r} --attention-dropout-p {attn} --ff-dropout-p {ff} --update-freq {uf} --reset-if-stuck --use-smaller-minibatch --logging --train-percent {args.train_percent}'
                        if args.dropout_decay:
                            cmd += ' --dropout-decay 1'
                        cmds.append(cmd)

with open(f'{args.script_name}.sh', 'w') as f:
    f.write('!#/bin/bash\n')
    f.write(f'echo "----------{args.script_name}-----------" > success.log\n')
    f.write(f'echo "----------{args.script_name}-----------" > error.log\n')
    for cmd in cmds:
        f.write(f'''
SECONDS=0
{cmd}
if [ $? -eq 0 ]; then
    echo "{cmd} took $SECONDS seconds"
    echo "{cmd} took $SECONDS seconds" >> success.log
else
    echo "{cmd}" >> error.log
fi\n''')
    f.write(f'echo "----------{args.script_name} end-----------" > success.log\n')
    f.write(f'echo "----------{args.script_name} end-----------" > error.log\n')
        
print(f'Generated {cnt} commands in {args.script_name}.sh')