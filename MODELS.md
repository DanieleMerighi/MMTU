# 🤖 Available Models

This file lists all pre-configured models available for inference.

## 📋 Registered Models

### 1. **qwen3-8b-awq** (Default)
- **HuggingFace ID**: `Qwen/Qwen3-8B-AWQ`
- **Size**: ~4.5 GB (quantized)
- **Speed**: ⚡⚡ Fast
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Best for**: Complex table tasks, high accuracy needed
- **Note**: Requires write permissions on `/llms/.locks/`

### 2. **qwen3-4b**
- **HuggingFace ID**: `Qwen/Qwen3-4B`
- **Size**: ~8 GB
- **Speed**: ⚡⚡⚡ Very Fast
- **Quality**: ⭐⭐⭐ Good
- **Best for**: Quick tests, iterative development
- **Note**: Good permissions, works out of the box

### 3. **llama-3.2-3b**
- **HuggingFace ID**: `meta-llama/Llama-3.2-3B-Instruct`
- **Size**: ~6 GB
- **Speed**: ⚡⚡⚡ Very Fast
- **Quality**: ⭐⭐⭐ Good
- **Best for**: General tasks, baseline comparisons
- **Note**: Good permissions, proven to work

### 4. **llama-3.1-8b-awq**
- **HuggingFace ID**: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`
- **Size**: ~4.5 GB (quantized)
- **Speed**: ⚡⚡ Fast
- **Quality**: ⭐⭐⭐⭐ Excellent
- **Best for**: Large-scale inference, good balance

## 🚀 Usage Examples

### Using pre-configured model names:
```bash
# Use Qwen3-8B-AWQ (default)
python3 inference.py -i test.jsonl self_deploy

# Use Qwen3-4B (faster)
python3 inference.py -i test.jsonl self_deploy --model qwen3-4b

# Use Llama 3.2 3B
python3 inference.py -i test.jsonl self_deploy --model llama-3.2-3b
```

### Using custom HuggingFace model IDs:
```bash
# Use any model from HuggingFace directly
python3 inference.py -i test.jsonl self_deploy --model "google/gemma-3-4b-it"
python3 inference.py -i test.jsonl self_deploy --model "Qwen/Qwen2.5-1.5B-Instruct"
```

### With Docker:
```bash
# Edit run_docker.sh to change the --model flag
# Or pass it as environment variable
MODEL=qwen3-4b sbatch run_docker.sh test.jsonl
```

## ➕ Adding New Models

To add a new model to the registry, edit `inference.py`:

```python
def load_your_model(max_new_tokens=2048):
    """Load Your Custom Model"""
    model_id = "organization/model-name"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        cache_dir="/llms"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir="/llms"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe.max_new_tokens = max_new_tokens
    return pipe, tokenizer

# Add to MODEL_REGISTRY
MODEL_REGISTRY = {
    ...
    "your-model-name": load_your_model,
}
```

## 🔍 Checking Available Models

```bash
python3 inference.py self_deploy --help
```

## 📊 Performance Comparison

| Model | Size | Speed | Accuracy | Memory | Permissions |
|-------|------|-------|----------|--------|-------------|
| qwen3-8b-awq | 4.5GB | ⚡⚡ | ⭐⭐⭐⭐ | ~8GB | ⚠️ Issues |
| qwen3-4b | 8GB | ⚡⚡⚡ | ⭐⭐⭐ | ~10GB | ✅ Good |
| llama-3.2-3b | 6GB | ⚡⚡⚡ | ⭐⭐⭐ | ~8GB | ✅ Good |
| llama-3.1-8b-awq | 4.5GB | ⚡⚡ | ⭐⭐⭐⭐ | ~8GB | ✅ Good |

## 💡 Tips

1. **Development**: Use `qwen3-4b` or `llama-3.2-3b` for quick iterations
2. **Production**: Use `qwen3-8b-awq` or `llama-3.1-8b-awq` for best quality
3. **Memory constraints**: Use AWQ models (they use ~50% less memory)
4. **Permission issues**: Stick to models with good permissions (qwen3-4b, llama-3.2-3b)
