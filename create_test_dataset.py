"""
Script per creare un subset strategico del dataset MMTU
Focus su task rilevanti per MCP e analisi table size
"""

from datasets import load_dataset
import json
from collections import defaultdict

print("📥 Caricamento dataset MMTU...")
ds = load_dataset("MMTU-benchmark/MMTU", split="train")
print(f"Dataset caricato: {len(ds)} esempi totali\n")

# Task MCP-relevant
# Focus su: NL2SQL, Table QA, Data Transform, Join, Data Imputation
MCP_RELEVANT_TASKS = [
    'NL2SQL',                              # NL to SQL conversion
    'Table-QA',                            # Table Question Answering
    'Data-transform-pbe',                  # Data transformation
    'Data-transform-reshape',              # Data transformation
    'equi-join-detect',                    # Table join detection
    'semantic-join',                       # Semantic join
    'Data-Imputation',                     # Data cleaning/imputation
    'Entity-Matching',                     # Entity matching/deduplication
    'Error-Detect',                        # Data cleaning
    'Schema-Matching',                     # Schema understanding
]

print("🎯 Task MCP-relevant selezionati:")
for task in MCP_RELEVANT_TASKS:
    print(f"  - {task}")
print()

# Filtra solo task MCP-relevant
print("🔍 Filtraggio task...")
mcp_data = ds.filter(lambda x: x['task'] in MCP_RELEVANT_TASKS)
print(f"Esempi filtrati: {len(mcp_data)}\n")

# Analizza distribuzione per task
task_distribution = defaultdict(int)
for example in mcp_data:
    task_distribution[example['task']] += 1

print("📊 Distribuzione esempi per task:")
for task, count in sorted(task_distribution.items()):
    print(f"  {task:40s} : {count:5d} esempi")
print()


print("Creazione subset strategico (max 100 esempi per task)...")

subset_examples = []
examples_by_task = defaultdict(list)

# Raggruppa esempi per task
for example in mcp_data:
    examples_by_task[example['task']].append(example)

# Prendi max 100 esempi per task
for task, examples in examples_by_task.items():
    n_samples = min(100, len(examples))
    subset_examples.extend(examples[:n_samples])
    print(f"  {task:40s} : {n_samples:3d} esempi selezionati")

print(f"\nSubset totale: {len(subset_examples)} esempi")

# Salva il subset
output_file = "mcp_strategic_subset.jsonl"
print(f"\n💾 Salvataggio in {output_file}...")

with open(output_file, 'w') as f:
    for example in subset_examples:
        f.write(json.dumps(example) + '\n')

print(f"Salvato! File: {output_file}")

# Statistiche aggiuntive
print("\n" + "="*60)
print("STATISTICHE SUBSET")
print("="*60)
print(f"Totale esempi: {len(subset_examples)}")
print(f"Numero task: {len(examples_by_task)}")
print(f"Media esempi per task: {len(subset_examples)/len(examples_by_task):.1f}")
print()

# Analizza dimensioni tabelle (se disponibile nei metadata)
print("🔍 Analizzando dimensioni tabelle...")
try:
    sizes = []
    for ex in subset_examples[:100]:  # Sample per non rallentare troppo
        metadata = json.loads(ex.get('metadata', '{}'))
        # Cerca informazioni su righe/colonne nei metadata
        # Questo dipende da come sono strutturati i metadata MMTU
        if 'rows' in metadata and 'columns' in metadata:
            sizes.append(metadata['rows'] * metadata['columns'])
    
    if sizes:
        print(f"Campione analizzato: {len(sizes)} tabelle")
        print(f"Dimensione media: {sum(sizes)/len(sizes):.0f} celle")
        print(f"Dimensione min: {min(sizes)} celle")
        print(f"Dimensione max: {max(sizes)} celle")
except Exception as e:
    print(f"Analisi dimensioni tabelle non disponibile: {e}")

print("\nDONE! Puoi ora eseguire inference su questo subset.")
print(f"Comando: python3 inference.py self_deploy --input {output_file}")
