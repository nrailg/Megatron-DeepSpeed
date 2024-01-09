data_path=./alpaca_data.json 
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

hf_llama_path=/data/llama-7b/
# weights link: https://huggingface.co/huggyllama/llama-7b

mega_ds_llama_path=/data/llama-7b-mega-ds-T2P2D2
# dir for mega_ds_weights

master_addr=198.10.1.11
# address of master node
master_port=5566

covert_args="deepspeed --hostfile hostfile \
	--master_addr $master_addr \
	--master_port $master_port \
	--num_nodes 2 --num_gpus 8 \
	hf2megads_weight_converter.py \
	--hf-ckpt-num-shards 2 \
	--origin-hf-ckpt-dir $hf_llama_path \
	--save $mega_ds_llama_path"

finetune_args="deepspeed --hostfile hostfile \
	--master_addr $master_addr \
	--master_port $master_port \
	--num_nodes 2 --num_gpus 8 \
	finetune_llama.py \
	--load $mega_ds_llama_path"

comm_args="--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 2 \
--lr-warmup-iters 2000 \
--weight-decay 0.1 \
--clip-grad 1 \
--num-layers 32 \
--hidden-size 4096 \
--num-attention-heads 32 \
--ffn-hidden-size 11008 \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--untie-embeddings-and-output-weights \
--swiglu \
--seq-length 512 \
--max-position-embeddings 512 \
--micro-batch-size 16 \
--global-batch-size 256 \
--train-iters 3500 \
--lr 2e-5 \
--tensorboard-dir tensorboard_output \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters 100 \
--eval-interval 100 \
--data-path $data_path \
--save-interval 1500 \
--split 100,0,0 \
--bf16 \
--zero-stage 0 \
--tokenizer-type HFTokenizer \
--tokenizer-model $hf_llama_path \
--deepspeed_config ./examples_deepspeed/finetune_hf_llama/ds_config.json \
--deepspeed \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--repeated-dataloader"

if [ "$1" = "convert" ]; then
    task_args="$covert_args"  
else
    task_args="$finetune_args" 
fi

full_cmd="$task_args $comm_args"

eval "$full_cmd"

