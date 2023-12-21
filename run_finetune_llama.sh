data_path=./alpaca_data.json 
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json


hf_llama_path=/host/mnt/disk2/kf/llama-7b
# weights link: https://huggingface.co/huggyllama/llama-7b

mega_ds_llama_path=./llama-7b-mega-ds-T2P2D2
# dir for mega_ds_weights

covert_args="deepspeed hf2megads_weight_converter.py \
--hf-ckpt-num-shards 2 \
--origin-hf-ckpt-dir $hf_llama_path \
--save $mega_ds_llama_path"

finetune_args="deepspeed finetune_llama.py \
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
--deepspeed_config ./examples_deepspeed/finetune_hf_llama \
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

