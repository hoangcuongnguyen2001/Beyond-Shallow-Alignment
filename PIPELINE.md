# Pipeline Guide

Step-by-step guide to reproducing the pipeline in this repository: safety
fine-tuning + mechanistic interpretability for three open-weight base models
— **Llama-3.1-8B**, **Gemma-2-9B**, and **Qwen3-8B**. Each model is
fine-tuned with several safety-training methods, then probed to find and
manipulate the internal "refusal direction," and finally re-evaluated on
jailbreak/over-refusal benchmarks to see how each method and intervention
shifts safety behavior.

For the research findings this pipeline produced, see [README.md](README.md).

The pipeline runs end-to-end in the order below — each stage reads the output
of the one before it.

## 0. Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch (CUDA 12.1), `transformers`, `trl`,
`peft`, `bitsandbytes`, `transformer_lens`, `sae-lens`, and `huggingface-hub`.

---

## 1. Download models — `download_models/`

Everything starts here. Base weights and the Llama-Guard judge model are
pulled from the Hugging Face Hub into a local `models/` directory.

```bash
python download_models/download_models.py   # currently pulls meta-llama/Llama-Guard-3-8B
```

Add your Hugging Face token (`login(token='...')`) before running, and add
entries for the base checkpoints you need (`Llama-3.1-8B`, `gemma-2-9b`,
`Qwen3-8B`) following the same `snapshot_download(...)` pattern.

## 2. Fix / add tokenizers — `download_models/`

Some checkpoints (fine-tuned outputs in particular) come back without a chat
template, or need one that doesn't match the base tokenizer's default. Run
these once per checkpoint, after download and after each training run that
produces a new checkpoint directory:

```bash
python download_models/add_tokenizer.py      # copies the reference chat template
                                               # (Llama-3.1-Instruct / Qwen3-8B / Gemma-2-9b-it)
                                               # onto any checkpoint missing one
python download_models/fix_chat_template.py   # hard-codes the canonical Llama-3 /
                                               # Gemma-2 Jinja templates onto specific
                                               # checkpoints (edit the `checkpoints` list)
```

Edit the `fixes` dict / `checkpoints` list in each script to point at the
checkpoint paths you actually have before running.

## 3. Train the safety variants — `training_scripts/`

Each base model is fine-tuned with several methods. Scripts are split by
method, then by model family:

| Directory | Method | Notes |
|---|---|---|
| `training_scripts/sft/` | Supervised fine-tuning | `train_sft_llama.py`, `train_sft_gemma.py`, `train_sft_qwen3.py` |
| `training_scripts/orpo/` | ORPO (preference optimization, no reference model) | `train_orpo_llama.py`, `train_orpo_gemma.py`, `train_orpo_qwen3.py` |
| `training_scripts/dpo/` | DPO / KTO | `train_dpo.py`, `train_kto_gemma.py`, `train_kto (1).py` |
| `training_scripts/ra-sft/` | Reasoning-augmented SFT (chain-of-thought) | `train_sft_cot_llama.py`, `train_sft_cot_gemma.py`, `train_sft_cot_qwen3.py` |
| `training_scripts/quantization_experiments/` | LoRA / QLoRA / OSFT variants | `train_sft_llama_lora.py`, `train_sft_llama_qlora.py`, `train_sft_gemma_qlora.py`, `train_sft_qwen3_qlora.py`, `train_sft_llama_osft.py`, `train_sft_osft.py` |

Datasets live in `training_datasets/`:

- `safety_sft_dataset.jsonl` — `{"instruction", "input", "output"}` (SFT)
- `safety_orpo_dataset.jsonl` — `{"prompt", "chosen", "rejected"}` (ORPO/DPO/KTO)
- `training_data.jsonl` — `{"instruction", "input", "output", "reasoning"}` (RA-SFT/CoT)

Preprocessing for the SFT and ORPO datasets is in
`training_scripts/Data_Preprocessing_SafetySFT.ipynb` and
`training_scripts/Dataset_Preprocessing_ORPO.ipynb`.

```bash
python training_scripts/sft/train_sft_llama.py
python training_scripts/orpo/train_orpo_llama.py
# ...repeat per model family / method
```

Shared conventions: `torch_dtype=torch.bfloat16` (never float16),
`attn_implementation="sdpa"`, effective batch size 128
(`per_device_train_batch_size=1`, `gradient_accumulation_steps=128`). Run
`download_models/add_tokenizer.py` again on each new output checkpoint before
moving on — the interpretability stages below load these checkpoints with
`transformer_lens`, which needs a working chat template.

## 4. Extract the refusal direction — `refusal_direction/`

`refusal_direction.py` (the "phase 1" script) loads each checkpoint
(base + every fine-tuned variant, configured in `MODEL_CONFIGS`), runs it
over paired harmful/harmless prompts, and extracts a per-layer "refusal
direction" vector (difference-of-means activation, à la ArDiti). It also
runs a layer-stability/candidate-layer sweep.

```bash
python refusal_direction/refusal_direction.py --model <key-from-MODEL_CONFIGS>
```

Uncomment the model entries you want in `MODEL_CONFIGS` first. Outputs land
in `refusal_direction/results/`:

- `directions_base.pt`, `directions_sft.pt`, `directions_orpo.pt`, `directions-ra-sft.pt` — the extracted direction vectors per model/layer
- `candidate_layers_magnitude.txt` — per-layer magnitude used to pick the intervention layer
- `llama_results.txt`, `gemma_results_new.txt`, `qwen3_results.txt` — run logs

## 5. Causal patching — `causal_patching/`

Validates that the extracted direction is actually causally responsible for
refusal, by patching activations along that direction and measuring the
effect on generations. Run against a specific checkpoint:

```bash
python causal_patching/activation_patching.py \
  --model_path <path/to/checkpoint> \
  --model_name <name> \
  --arch {llama|gemma|qwen3} \
  --harmful_path <prompts.jsonl> \
  --harmless_path <prompts.jsonl> \
  --output_dir results/phase2
```

Then, to attribute the effect to individual components (heads/layers) rather
than the whole residual stream:

```bash
python causal_patching/component_analysis.py --model_path ... --model_name ... --arch ... \
  --harmful_path ... --harmless_path ... --output_dir results/phase2

python causal_patching/component_patching.py --model_path ... --model_name ... --arch ... \
  --harmful_path ... --harmless_path ... --component_json <from component_analysis.py>

python causal_patching/bootstrap_component_patching.py --pt_path <component_patching.pt output> \
  --n_boot 1000 --top_k 5
```

Finally, plot everything:

```bash
python causal_patching/visualisation.py --results_dir results/phase2 --arch {llama|gemma|both} \
  --phase1_dir results/phase1 --output_dir plots/phase2
```

Model checkpoints used for this stage live under `causal_patching/models/`
(one subfolder per model × training-method combination).

## 6. Steering interventions — `steering/`

Uses the direction/heads identified above to actively steer generations at
inference time, via two methods, then evaluates the steered model on
held-out prompt sets in `steering/datasets/` (`wildjailbreak_250.jsonl`,
`xstest_safe.jsonl`, `mmlu_200.jsonl`).

**ActAdd** (`steering/actadd/`) — adds a scaled direction vector to the
residual stream at a chosen layer:

```bash
python steering/actadd/actadd_vectors.py --model_path ... --model_name ... \
  --direction_repo ... --direction_filename ... --peak_layer <int>

python steering/actadd/run_xstest_actadd.py --model_path ... --model_name ... \
  --vector_path ... --target_layers ... --alpha <float>

python steering/actadd/wildjailbreak_actadd.py --model_path ... --model_name ... \
  --vector_path ... --target_layers ... --judge_path meta-llama/Llama-Guard-3-8B

python steering/actadd/mmlu_check.py   # utility-preservation sanity check
```

**ITI — Inference-Time Intervention** (`steering/iti/`) — trains linear
probes on specific attention heads and steers along the probe direction:

```bash
python steering/iti/gemma_iti_pipeline.py --model_path ... --model_name ... \
  --target_heads 37:15 38:15 \
  --fit_harmful_prompts ... --fit_harmless_prompts ... \
  --eval_harmful_prompts ...

python steering/iti/run_xstest_iti.py --model_path ... --model_name ... \
  --probe_path <from gemma_iti_pipeline.py> --target_heads 37:15 38:15 --alphas 0 5 10 20 40

python steering/iti/wildjailbreak_iti.py --model_path ... --model_name ... \
  --probe_path ... --target_heads ... --alpha <float> \
  --judge_model_path ... --judge_tokenizer_path ...

python steering/iti/mmlu_iti.py --model_path ... --model_name ... \
  --probe_path ... --target_heads ... --alphas ... --mmlu_path steering/datasets/mmlu_200.jsonl
```

## 7. Refusal / safety profiling — `refusal_profile/`

Benchmarks any checkpoint (base, fine-tuned, or steered) for attack success
rate and over-refusal:

```bash
# Jailbreak attack success rate against a curated harmful prompt set
python refusal_profile/StrongREJECT/run_strongreject.py

# Jailbreak ASR on WildJailbreak, judged by Llama-Guard
python refusal_profile/WildJailbreak_XSTest/wildjailbreak.py --judge_path meta-llama/Llama-Guard-3-8B

# Over-refusal check on safe-but-scary-sounding prompts (XSTest)
python refusal_profile/WildJailbreak_XSTest/run_xstest.py

# Bootstrap confidence intervals over the above results
python refusal_profile/WildJailbreak_XSTest/bootstrap_asr.py \
  --strongreject <path(s)> --xstest <path(s)> --pairs <path(s)> --n_boot 10000
```

Results land under `refusal_profile/results/{StrongREJECT,WildJailbreak,XSTest}/`,
with aggregate significance in `bootstrap_results.txt`.

---

## Directory summary

| Directory | Role |
|---|---|
| `download_models/` | Stage 1–2: pull weights, fix/add tokenizers |
| `training_scripts/` | Stage 3: SFT / ORPO / DPO-KTO / RA-SFT / LoRA-QLoRA-OSFT training |
| `training_datasets/` | Datasets consumed by stage 3 |
| `refusal_direction/` | Stage 4: extract the refusal direction per model/checkpoint |
| `causal_patching/` | Stage 5: validate + localize the direction causally |
| `steering/` | Stage 6: ActAdd / ITI interventions + their eval datasets |
| `refusal_profile/` | Stage 7: jailbreak ASR (StrongREJECT, WildJailbreak) + over-refusal (XSTest) benchmarking |
