"""
Analisi delle dimensioni delle tabelle nel dataset MMTU
Questo script estrae e analizza le dimensioni delle tabelle per identificare soglie
"""

import json
import re
from collections import defaultdict
import os

print("="*70)
print("📊 ANALISI DIMENSIONI TABELLE - MMTU DATASET")
print("="*70)
print()

# Carica i risultati esistenti
result_file = "mmtu.self_deploy.result.jsonl"

if not os.path.exists(result_file):
    print(f"❌ File {result_file} non trovato!")
    print("   Esegui prima l'inference per generare i risultati.")
    exit(1)

print(f"📂 Caricamento risultati da: {result_file}")

examples = []
with open(result_file, 'r') as f:
    for line in f:
        examples.append(json.loads(line))

print(f"✅ Caricati {len(examples)} esempi\n")

# Funzione per estrarre dimensioni tabella dal prompt
def extract_table_dimensions(prompt_text):
    """
    Cerca di estrarre dimensioni tabella dal prompt.
    Le tabelle in markdown hanno header e righe.
    """
    if not prompt_text:
        return None, None
    
    # Cerca tabelle in formato markdown
    lines = prompt_text.split('\n')
    table_lines = [l for l in lines if l.strip().startswith('|') and l.strip().endswith('|')]
    
    if len(table_lines) >= 2:
        # Prima riga = header, seconda = separatore, resto = dati
        header = table_lines[0]
        # Conta le colonne (numero di | - 1, esclusi quelli esterni)
        cols = len([c for c in header.split('|') if c.strip()]) 
        # Conta le righe (escludi header e separatore)
        rows = len(table_lines) - 2  # -2 per header e separatore
        
        if rows > 0 and cols > 0:
            return rows, cols
    
    return None, None

# Analizza dimensioni per ogni esempio
print("🔍 Analizzando dimensioni tabelle...")
sizes_by_task = defaultdict(list)
all_sizes = []

for example in examples:
    task = example.get('task', 'Unknown')
    prompt = example.get('prompt', '')
    
    rows, cols = extract_table_dimensions(prompt)
    
    if rows and cols:
        size = rows * cols
        sizes_by_task[task].append({
            'rows': rows,
            'cols': cols,
            'cells': size
        })
        all_sizes.append(size)

print(f"✅ Analizzate {len(all_sizes)} tabelle\n")

# Statistiche globali
if all_sizes:
    print("="*70)
    print("📈 STATISTICHE GLOBALI")
    print("="*70)
    print(f"Totale tabelle analizzate: {len(all_sizes)}")
    print(f"Dimensione media: {sum(all_sizes)/len(all_sizes):.0f} celle")
    print(f"Dimensione mediana: {sorted(all_sizes)[len(all_sizes)//2]} celle")
    print(f"Dimensione min: {min(all_sizes)} celle")
    print(f"Dimensione max: {max(all_sizes)} celle")
    print()
    
    # Distribuzione per soglie
    thresholds = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    print("📊 Distribuzione per soglie dimensionali:")
    prev_threshold = 0
    for threshold in thresholds:
        count = len([s for s in all_sizes if prev_threshold <= s < threshold])
        percentage = (count / len(all_sizes)) * 100
        print(f"  {prev_threshold:6d} - {threshold:6d} celle: {count:4d} tabelle ({percentage:5.1f}%)")
        prev_threshold = threshold
    
    count = len([s for s in all_sizes if s >= prev_threshold])
    percentage = (count / len(all_sizes)) * 100
    print(f"  {prev_threshold:6d}+         celle: {count:4d} tabelle ({percentage:5.1f}%)")
    print()

# Statistiche per task
print("="*70)
print("📊 STATISTICHE PER TASK")
print("="*70)
for task in sorted(sizes_by_task.keys()):
    sizes = [s['cells'] for s in sizes_by_task[task]]
    if sizes:
        print(f"\n{task}:")
        print(f"  Esempi analizzati: {len(sizes)}")
        print(f"  Dimensione media: {sum(sizes)/len(sizes):.0f} celle")
        print(f"  Range: {min(sizes)} - {max(sizes)} celle")
        
        # Trova un esempio rappresentativo
        avg_size = sum(sizes)/len(sizes)
        closest_idx = min(range(len(sizes)), key=lambda i: abs(sizes[i] - avg_size))
        representative = sizes_by_task[task][closest_idx]
        print(f"  Esempio tipico: {representative['rows']} righe × {representative['cols']} colonne")

print()
print("="*70)
print("💡 SUGGERIMENTI PER LA TESI")
print("="*70)
print()
print("Basandoti su questa analisi, considera di testare le seguenti soglie:")
print()

if all_sizes:
    # Suggerisci soglie basate sui dati reali
    median = sorted(all_sizes)[len(all_sizes)//2]
    p75 = sorted(all_sizes)[int(len(all_sizes)*0.75)]
    p90 = sorted(all_sizes)[int(len(all_sizes)*0.90)]
    
    print(f"1. SMALL tables (< {median} celle)")
    print(f"   → Usa inclusione diretta della tabella nel prompt")
    print()
    print(f"2. MEDIUM tables ({median} - {p75} celle)")
    print(f"   → Usa approccio ibrido (schema + sample + SQL se necessario)")
    print()
    print(f"3. LARGE tables (> {p75} celle)")
    print(f"   → Usa solo SQL queries iterative (MCP tools)")
    print()
    print(f"4. VERY LARGE tables (> {p90} celle)")
    print(f"   → Richiede ottimizzazioni speciali")

print()
print("✅ Analisi completata!")
print()
print("📝 Prossimi passi per la tesi:")
print("   1. Esegui inference con modello migliore")
print("   2. Confronta performance tra soglie dimensionali")
print("   3. Crea grafici accuracy vs table size")
print("   4. Identifica threshold ottimale per switching approcci")
