import pandas as pd
import json
import os


MODEL_VARIANTS = {
    "safety-sft": "/workspace/models/llama-sft",
    "safety-orpo": "/workspace/models/llama-orpo",
    "safety-kto": "/workspace/models/llama-kto",
    "safety-ra-sft": "/workspace/models/llama-ra-sft",  # add when ready
}

RESPONSES_DIR = "/workspace/responses"
SCORES_DIR = "/workspace/scores"

def compute_asr(scores_path):
    with open(scores_path) as f:
        scores = json.load(f)

    df = pd.DataFrame(scores)

    overall = df['asr_score'].mean()
    by_jailbreak = df.groupby('jailbreak')['asr_score'].mean()
    by_category = df.groupby('category')['asr_score'].mean()

    return overall, by_jailbreak, by_category


print("\n" + "="*60)
print("ASR RESULTS")
print("="*60)

heatmap_rows = []

for model_name in MODEL_VARIANTS.keys():
    scores_path = os.path.join(SCORES_DIR, f"{model_name}_scores.json")

    if not os.path.exists(scores_path):
        print(f"No scores found for {model_name} — skipping")
        continue

    overall, by_jailbreak, by_category = compute_asr(scores_path)

    print(f"\n{model_name}:")
    print(f"  Overall ASR: {overall:.3f}")
    print(f"  By jailbreak type:")
    for jb, asr in by_jailbreak.items():
        print(f"    {jb}: {asr:.3f}")

    for jb, asr in by_jailbreak.items():
        heatmap_rows.append({
            "model": model_name,
            "jailbreak": jb,
            "asr": asr
        })

# Heatmap for paper
df_heatmap = pd.DataFrame(heatmap_rows)

if not df_heatmap.empty:
    heatmap = df_heatmap.pivot(
        index='jailbreak',
        columns='model',
        values='asr'
    ).round(3)

    print("\nHeatmap (rows=jailbreak type, cols=model variant):")
    print(heatmap)

    heatmap.to_csv(
        os.path.join(SCORES_DIR, "asr_heatmap.csv")
    )
    print(f"\nHeatmap saved to {SCORES_DIR}/asr_heatmap.csv")