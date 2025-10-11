from datasets import load_dataset, concatenate_datasets

SEED = 42
K = 5  # samples per task

print("Loading MMTU dataset...")
ds = load_dataset("MMTU-benchmark/MMTU", split="train")

print(f"Total samples: {len(ds)}")
print(f"Creating test dataset with {K} samples per task...")

subs = []
for t in sorted(set(ds["task"])):
    d = ds.filter(lambda x: x["task"] == t).shuffle(seed=SEED)
    subs.append(d.select(range(min(K, len(d)))))

final = concatenate_datasets(subs)
print(f"Final test dataset: {len(final)} samples")

output_file = "mmtu.jsonl"
final.to_json(output_file, lines=True, force_ascii=False)
print(f"Dataset saved to {output_file}")