Training LLMs:
    3 approaches to finetune LLM

    1. Pretraining:
        Source:
            1. DATA:
                Massive text data, wikipedia, data in TBs, extracting and creating 2 trillion tokens. In the pretaining step we have massive text data in terabytes.
            2. MODEL ARCHITECTRURE:
                Identify the model architecuture for task at hand.
            3. TOKENIZER:
                Need a tokenize that is trained to handle the data.  Ensuring it can 
               efficiently encode and decode text.
            4. PREPROCESS DATA:
                Preprocess data using tokenizer vocabulary using library like SentencePiece.  Convert raw text into a format suitable to train a model.
                1. Mapping tokens to corresponding IDs.
                2. Incorporating any special token
                3. Attention masks
            5. PRETRAINING PHASE:
                Model learns to predict the next word in the sentence.
                Filling missing words
                Involve optimize model parameters to maximize likelyhood of generating
                sequence of words given the context.
                We apply self supervised learning techniques
                Predict missing tokens based on the surrounding context.
                Model understand the language pattern, grammar and semantic relationships.
                1. Fill in missing words:
                    Masked language model.
                2. Text Generating:
                    Causal language modeling.
            At this stage it captured general language but it lacks specific knowledge about a particular task or domain

    2. Finetuning:
        We want to finetune for a specific purpose.  Allows specialize LLMs capability and optimize its performance for the narrower downstream tasks.
        Fine tuning is done to minimize task specific loss function.
        Parameters of the pretrained are "adjusted/adjusting weights" using gradient based optimization algorithms like SGD/Adam etc.
        Gradients are created by backpropogating loss.  Model learns and updates the parameters.
        Finetuning using 
            Learning rate scheduling
            regularization methods like dropouts, weight decay, early stopping to prevent overfitting.
            Finetune model should generalized on unseen data.
        Instruct Tuning/Instruction Tuned Model:
            Dataset of instruction  and response is gathered.
            Here we need dataset of instruction and response.
            Here we need lower number of sample data 10K data is fine.
    3. LoRA/QLoRA
        These can be used to finetune
        LoRA: Low Rank Adaptation:
            FT is expensive.  It is giving 3X reduction in memory requirement
        QLoRA: Quantized LoRA:
            It uses a library called bitsandbytes.  It achieves near lossless quantization.  This reduces memory requirment for FT.
            From 16x A100 [80 GB GPU] to 2x RTX 3090 or 1 A100
    
FineTuning:
    [6B/7B LLM]
    Training Compute:
        To finetune LLaMA-2 or Mistral These models can be commercially used.
    Memory:
        150-195 GB
    
    Renting GPU:
        1. Runpod.io (Preferred Choice)
        2. Google Collab
        3. VastAI
        4. Lambda Labs
        5. AWS Sagemaker
    
Gather Dataset:
    Main source Huggingface
        Diversity in data
        Size of data (Filesize should be 10 MB of data, 10K question answer pairs)
        Quality of data

To finetune:
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
Examples folder has configuration used with different model
We need to change the dataset here.
Path to dataset looks like following:
datasets:
  - path: teknium/GPT4-LLM-Cleaned
    type: alpaca
pip install packaging
pip install -e '.[flash_attn,deepspeed]'
Install pytorch
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install py-cpuinfo
pip install psutil
Install requirements.txt

The release on PyPI should work with the following assumptions about your environment:

pip install deepspeed-kernels
NVIDIA GPU(s) with compute capability of: 8.0, 8.6, 8.9, 9.0
PyPI release on A100, A6000, and H100.
CUDA 11.6+
Ubuntu 20+

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

### Usage
```bash
# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/openllama-3b/lora.yml

# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./lora-out"

# gradio
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./lora-out" --gradio

# remote yaml files - the yaml config can be hosted on a public URL
# Note: the yaml config must directly link to the **raw** yaml
accelerate launch -m axolotl.cli.train https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/examples/openllama-3b/lora.yml
```

LoRA (Memory Efficiency)
    Training method designed to expedite training of LLMs
    It reduces memory consumption by introducing pair of rank-decomposition weight matrices 
    1. Preservation of pretrained weights: LoRA maintains the frozen state of previously trained weights.  It minimize the risk of forgetting.  This ensures that model retains its existing knowledge while adapting to new data.
    2. Portability of the trained weight. rank matrics used with LoRA has significantly lower number of parameters.  This allows trained LoRA weights to be utilitized with other context.
    3. Integration with attention layers: LoRA matrics are incorporated in the attention layer of original model.  Allows extend to which model adapts to the new data.


LoRAs (Hyper Parameters From lora.yaml file)
    lora_r: 
        (LoRA rank, determines number of rank decomposition matrices, rank decomponsition is applied to reduce memory consumption and computation requirements. Higher rank better results but require higher compute)
    lora_alpha:
        (Scaling factor that determines the extend to which model adapts to new value.Alpha value contributes to update of weights during training. Lower value gives more weight to the original data. It maintains model's existing knowledge to greater extend)
    lora_dropout: 0.0
    lora_target_modules:
        (Determine which specific weight or matrices we want to train, Query vector (applies to the query of transform block) and Value vectors q_proj and v_proj (transform the hidden state to effective query or value dimensions) )
    - gate_proj
    - down_proj
    - up_proj
    - q_proj
    - v_proj
    - k_proj
    - o_proj (output of attention)
    lora_fan_in_fan_out:

    To get projection:
        layer_names = model.state.dict().keys()

Get hidden_size from hugging face model's config.json file

lm head (output layer of LM)
embed_token

QLoRA:
    Quantized LoRA
    backprop of gradient thru frozed 4 bit quantized into LoRA
    handle normally distributed weights

batch_size
num_epochs
lr_scheduler
optimizer
wanddb == Weights and biases




    

        






    


