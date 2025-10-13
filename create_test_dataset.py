# 1. Esegui il pipeline
from datasets import load_dataset
ds = load_dataset("MMTU-benchmark/MMTU", split="train")

# 2. Filtra solo task MCP-relevant
mcp_tasks = ['table-join', 'equi-join', 'column-transform', 
             'data-cleaning', 'data-imputation', 'nl-to-sql']
mcp_data = ds.filter(lambda x: x['task'] in mcp_tasks)

# 3. Prendi 50 esempi per task per test rapido
small_test = mcp_data.select(range(min(300, len(mcp_data))))

# 4. Salva e testa
small_test.to_json("mcp_test.jsonl", lines=True)