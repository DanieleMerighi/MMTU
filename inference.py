import pandas as pd
import openai
import os
import sys
import random
from io import StringIO
import threading
import queue
import json
from datetime import date
from openai import AzureOpenAI, OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_random,
    wait_fixed,
    RetryError
)  # for exponential backoff
from tqdm import tqdm
import time
import urllib.request
import re
import shutil

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport


import logging
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logformatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logformatter)
logger.addHandler(consoleHandler)


# Queue to hold queries
query_queue = queue.Queue()

# Lock for synchronizing file write operations
file_lock = threading.Lock()

@retry(wait=wait_random_exponential(max=30, multiplier=2), stop=stop_after_attempt(6))
def query_request(client, model, prompt, temperature=1.0, timeout=90):
    response = ""
    prompt_token = None
    completion_token = None
    time_taken = None
    t0 = time.time()
    try:
        # print(f"Start time: {time.time()}")
        completion = client.with_options(timeout=timeout).chat.completions.create(
            model=model, messages=[
                # {"role": "system", "content": "You are an AI assistant. Provide an answer to some data/table related questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            timeout=timeout,
            # request_timeout=timeout,
        )
        t1 = time.time()
        logger.debug(completion.usage.model_dump_json())
        response = completion.choices[0].message.content
        prompt_token = completion.usage.prompt_tokens
        completion_token = completion.usage.completion_tokens
        time_taken = t1 - t0
        return {
            "response": response,
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "time_taken": time_taken
        }
    except openai.NotFoundError as e:
        logger.error(f"Deployment {model} not found")
        exit(1)
    except openai.APITimeoutError as e:
        t1 = time.time()
        logger.error(f"Timeout error (timeout={timeout}): {e}")
        response = f"API Timeout (timeout={timeout}) (timeout at {t1-t0:.2f} seconds)"
        time_taken = t1 - t0
        return {
            "response": response,
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "time_taken": time_taken
        }
    except openai.RateLimitError as e:
        logger.error(f"Rate limit error.")
        raise e
    except openai.InternalServerError as e:
        logger.error(f"Internal server error.")
        raise e
        # exit(1)
    except Exception as e:
        # print(f"End time: {time.time()}")
        # logger.error(f"Error processing query: {e}")
        print(e.__dict__)
        if hasattr(e, 'code'):
            logger.error(
                    f"Error processing query: {e.code}") # type: ignore
            if e.code == "context_length_exceeded": # type: ignore
                response = "Error: string above max length"
                t1 = time.time()
                return {
                    "response": response,
                    "prompt_tokens": prompt_token,
                    "completion_tokens": completion_token,
                    "time_taken": t1 - t0
                }
            elif e.code == 424: # type: ignore
                pass
            elif e.code == "content_filter":
                response = "Error: content filter"
                t1 = time.time()
                return {
                    "response": response,
                    "prompt_tokens": prompt_token,
                    "completion_tokens": completion_token,
                    "time_taken": t1 - t0
                }
            else:
                raise e
        else:
            print(e)
            logger.error(f"Error processing query: {e}")
        if hasattr(e, 'message'):
            if e.message == "Connection error.":  # type: ignore
                logger.error(f"Connection error.")
                exit(1)
        raise e
    
@retry(wait=wait_random_exponential(max=30, multiplier=2), stop=stop_after_attempt(6))
def query_request_ai_foundry(client, model, prompt, temperature=1.0, timeout=90):
    response = ""
    prompt_token = None
    completion_token = None
    time_taken = None
    t0 = time.time()
    try:
        completion = client.complete(
            messages=[
                UserMessage(content=prompt)
            ],
            temperature=temperature,
            model=model,
            timeout=timeout,
        )
        t1 = time.time()
        logger.debug(completion.usage)
        response = completion.choices[0].message.content
        prompt_token = completion.usage.prompt_tokens
        completion_token = completion.usage.completion_tokens
        time_taken = t1 - t0
        return {
            "response": response,
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "time_taken": time_taken
        }
    except Exception as e:
        # print(f"End time: {time.time()}")
        # logger.error(f"Error processing query: {e}")
        t1 = time.time()
        # print(e.__dict__)
        if hasattr(e, 'code'):
            logger.error(
                    f"Error processing query: {e.code}") # type: ignore
            if e.code == "context_length_exceeded": # type: ignore
                response =  "Error: string above max length"
                return {
                    "response": response,
                    "prompt_tokens": prompt_token,
                    "completion_tokens": completion_token,
                    "time_taken": t1 - t0
                }
            elif e.code == 424: # type: ignore
                pass
            elif e.code == 408: # type: ignore
                logger.error(f"Timeout error (timeout={timeout}): {e}")
                response = f"API Timeout (timeout={timeout}) (timeout at {t1-t0:.2f} seconds)"
                return {
                    "response": response,
                    "prompt_tokens": prompt_token,
                    "completion_tokens": completion_token,
                    "time_taken": t1 - t0
                }
            else:
                logger.error(f"Error processing query: {e}")
                # raise e
        else:
            print(e)
            logger.error(f"Error processing query: {e}")
        if hasattr(e, 'message'):
            if e.message == "Connection error.":  # type: ignore
                logger.error(f"Connection error.")
                exit(1)
            elif e.message.find("content_filter") != -1:
                response =  "Error: content filter"
                return {
                    "response": response,
                    "prompt_tokens": prompt_token,
                    "completion_tokens": completion_token,
                    "time_taken": t1 - t0
                }
            elif e.message.find("timeout") != -1 or e.message.find("Timeout") != -1:
                t1 = time.time()
                logger.error(f"Timeout error: {e}")
                response = f"API Timeout (timeout={timeout}) (timeout at {t1-t0:.2f} seconds)"
                return {
                    "response": response,
                    "prompt_tokens": prompt_token,
                    "completion_tokens": completion_token,
                    "time_taken": t1 - t0
                }
        raise e
    
def create_query_funtion_openai(endpoint, model, api_key, api_version):
    def query(prompt, temperature):
        client = OpenAI(
            api_base=endpoint,
            api_version=api_version,
            api_key=api_key
        )
        return query_request(client, model, prompt, temperature)
    query.__name__ = f"query_with_{model.replace('-', '_')}"
    return query

def create_query_funtion_azure_openai(endpoint, model, api_key, api_version):
    def query(prompt, temperature):
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        return query_request(client, model, prompt, temperature)
    query.__name__ = f"query_with_{model.replace('-', '_')}"
    return query

def create_query_function_ai_foundry(endpoint, model, api_key):
    def query(prompt, temperature):
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
            transport=RequestsTransport(read_timeout=90)
        )
        return query_request_ai_foundry(client, model, prompt, temperature)
    query.__name__ = f"query_with_{model.replace('-', '_')}"
    return query

def query_worker(one_query_function_pointer, temperature, output_file, pbar, model_name="OpenAI"):
    while not query_queue.empty():
        json_line = None
        result = None
        
        response = ""
        prompt_token = None
        completion_token = None
        time_taken = None
        
        try:
            json_line = query_queue.get()
            messages = json_line['prompt']
            result = one_query_function_pointer(messages, temperature)
            logger.info(
                f"processed with {one_query_function_pointer.__name__}")
        except queue.Empty:
            break
        except Exception as e:
            print(f"Failed query: {e}")
            response = ""
        finally:
            query_queue.task_done()
        if json_line:
            if type(result) == str:
                response = result
            elif type(result) == dict:
                assert "response" in result
                response = result.get("response", "")
                prompt_token = result.get("prompt_tokens", None)
                completion_token = result.get("completion_tokens", None)
                time_taken = result.get("time_taken", None)
            elif result is None:
                response = ""
            else:
                raise ValueError(f"Invalid result type: {type(result)}")
            json_line['response'] = response
            json_line['prompt_tokens'] = prompt_token
            json_line['completion_tokens'] = completion_token
            json_line['time_taken'] = time_taken
            json_line['model_name'] = model_name
            line = json.dumps(json_line) + "\n"

            pbar.update(1)

            # Safely write response to a file
            with file_lock:
                with open(output_file, 'a') as file:
                    file.write(line)

# main entry point for multi-thread querying


def query_chat_endpoint(input_file, output_file, query_endpoints, temperature, n_parallel_call_per_key=4, shuffle=False, model_name="OpenAI"):
    print(f"Evaluating {input_file} ...")
    t0 = time.time()
    all_queries = pd.read_json(input_file, lines=True)
    query_list = []
    if os.path.exists(output_file):
        print(f"Start reading output file {output_file} ...")
        cur_rst = pd.read_json(output_file, lines=True)
        if len(cur_rst) == 0:
            query_list = all_queries.to_dict(orient='records')
        else:
            failed_queries = cur_rst[cur_rst['response'] == ""]
            query_list = all_queries[(~all_queries['metadata'].isin(cur_rst['metadata'])) | all_queries['metadata'].isin(failed_queries['metadata'])].to_dict(orient='records')
            cur_rst[cur_rst['response'] != ""].to_json(output_file, orient='records', lines=True)
            print(f"Output file contains {len(cur_rst) - len(failed_queries)} queries.")
            print(f"# of TODO queries: {len(query_list)} out of {len(all_queries)}")
            
            cur_rst = pd.read_json(output_file, lines=True)
        
            if not len(all_queries) == len(cur_rst) + len(query_list): #, f"len(all_queries)={len(all_queries)}, len(cur_rst)={len(cur_rst)}, len(failed_queries)={len(failed_queries)}, len(query_list)={len(query_list)}"
                print(f"Error: {len(all_queries)} != {len(cur_rst) + len(query_list)}")
                os.remove(output_file)
                query_list = all_queries.to_dict(orient='records')
                with open(output_file, 'w') as f:
                    pass  # create empty file
    else:
        query_list = all_queries.to_dict(orient='records')
        with open(output_file, 'w') as f:
            pass  # create empty file
        
    
    if len(query_list) == 0:
        print("All queries are already processed.")
        return

    # query_list = []
    # with open(input_file, 'r') as f:

    #     for line in f:
    #         # Parse each line as JSON and add to the list
    #         json_line = json.loads(line)

    #         # add all queries to queue
    #         # query_queue.put(json_line)
    #         query_list.append(json_line)
    if shuffle:
        print("Shuffle: True")
        random.shuffle(query_list)
        
    for json_line in query_list:
        query_queue.put(json_line)

    # mutli-threaded querying
    all_query_function_pointers = query_endpoints*n_parallel_call_per_key
    threads = []

    total_tasks = query_queue.qsize()
    progress_bar = tqdm(total=total_tasks, desc=f"Querying {model_name}", ncols=100)

    for one_query_function_pointer in all_query_function_pointers:
        thread = threading.Thread(target=query_worker, args=(
            one_query_function_pointer, temperature, output_file, progress_bar, model_name))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    progress_bar.close()
    t1 = time.time()
    print("All threads completed\n")
    # print the time with two decimal places
    print(f"Time taken: {t1-t0:.2f} seconds")
    
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

_TOK=None; _PIPE=None; _CURRENT_MODEL=None

# ============================================================================
# MODEL LOADING FUNCTIONS - Add your models here!
# ============================================================================

def load_qwen3_8b_awq(max_new_tokens=2048):
    """Load Qwen3-8B-AWQ (quantized, efficient)"""
    model_id = "Qwen/Qwen3-8B-AWQ"
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

def load_qwen3_4b(max_new_tokens=2048):
    """Load Qwen3-4B (faster, smaller)"""
    model_id = "Qwen/Qwen3-4B"
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

def load_llama_3_2_3b(max_new_tokens=2048):
    """Load Llama 3.2 3B Instruct"""
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
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

def load_llama_3_1_8b_awq(max_new_tokens=2048):
    """Load Llama 3.1 8B AWQ (quantized)"""
    model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
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

def load_custom_model(model_id, max_new_tokens=2048):
    """Load any HuggingFace model by ID"""
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

def load_qwen2_5_7b(max_new_tokens=2048):
    """Load Qwen2.5-7B-Instruct (excellent for structured reasoning and tables)"""
    model_id = "Qwen/Qwen2.5-7B-Instruct"
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

def load_qwen2_5_14b(max_new_tokens=2048):
    """Load Qwen2.5-14B-Instruct (BEST for MMTU - powerful reasoning, table understanding)"""
    model_id = "Qwen/Qwen2.5-14B-Instruct"
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

# ============================================================================
# MODEL REGISTRY - Map model names to loading functions
# ============================================================================

MODEL_REGISTRY = {
    "qwen2.5-14b": load_qwen2_5_14b,
    "qwen2.5-7b": load_qwen2_5_7b,
    "qwen3-8b-awq": load_qwen3_8b_awq,
    "qwen3-4b": load_qwen3_4b,
    "llama-3.2-3b": load_llama_3_2_3b,
    "llama-3.1-8b-awq": load_llama_3_1_8b_awq,
}

def _lazy_load(model_name="qwen3-8b-awq", max_new_tokens=2048):
    """
    Lazy load a model by name.
    
    Args:
        model_name: Short name from MODEL_REGISTRY or full HuggingFace model ID
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        tuple: (pipeline, tokenizer)
    """
    global _TOK, _PIPE, _CURRENT_MODEL
    
    # If model is already loaded and same as requested, return it
    if _PIPE and _CURRENT_MODEL == model_name:
        return _PIPE, _TOK
    
    print(f"Loading model: {model_name}")
    
    # Check if it's a registered model
    if model_name in MODEL_REGISTRY:
        _PIPE, _TOK = MODEL_REGISTRY[model_name](max_new_tokens)
    else:
        # Assume it's a full HuggingFace model ID
        print(f"Model not in registry, loading as custom: {model_name}")
        _PIPE, _TOK = load_custom_model(model_name, max_new_tokens)
    
    _CURRENT_MODEL = model_name
    return _PIPE, _TOK

def self_deploy_query_function(model_name="qwen3-8b-awq"):
    """
    Create a query function for a specific model.
    
    Args:
        model_name: Model name from registry or HuggingFace ID
    """
    _lazy_load(model_name)
    def query(prompt, temperature=0.0):
        pipe, tok = _lazy_load(model_name)
        chat = tok.apply_chat_template(
            [{"role":"user","content":prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        out = pipe(
            chat, 
            max_new_tokens=pipe.max_new_tokens, 
            do_sample=(temperature>0),
            temperature=(float(temperature) if temperature>0 else None),
            return_full_text=False, 
            eos_token_id=tok.eos_token_id
        )[0]["generated_text"]
        return out
    return query

def main(args):
    if args.api_provider == "openai":
        query_func = create_query_funtion_openai(
            args.endpoint, args.model, args.api_key, args.api_version)
        query_endpoints = [query_func]
    elif args.api_provider == "azure_openai":
        query_func = create_query_funtion_azure_openai(
            args.endpoint, args.model, args.api_key, args.api_version)
        query_endpoints = [query_func]
    elif args.api_provider == "azure_ai_foundry":
        query_func = create_query_function_ai_foundry(
            args.endpoint, args.model, args.api_key)
        query_endpoints = [query_func]
    elif args.api_provider == "self_deploy":
        query_func = self_deploy_query_function(args.model)
        query_endpoints = [query_func]
    else:
        raise ValueError("Invalid API provider")
    
    if len(args.input_file) == 0:
        logger.setLevel(logging.DEBUG)
        # test the query function
        test_input = "What is the capital of France?"
        for query_func in query_endpoints:
            print("="*50)
            print(f"Test function: {query_func.__name__}")
            print(f"Model name: {args.model}")
            print(f"Test input: {test_input}")
            try:
                test_output = query_func(test_input, args.temperature)
                print(f"Funtion returned: {test_output}")
                assert "response" in test_output, "Response not found in output"
                print(f"Response: {test_output['response']}")
                print("Test Passed.")
            except Exception as e:
                print(f"Error: {e}")
                print("Test Failed.")
                exit(1)
        logger.setLevel(logging.WARN)

    if args.input_file:
      temp = args.temperature if args.temperature is not None else 0.0
      # Determina il model_name in base al provider
      model_name = getattr(args, 'model', args.api_provider)
      for input_file in args.input_file:
        assert os.path.exists(input_file), f"{input_file} not found"
        # nome output: <input>.<model_name>.result.jsonl
        base = os.path.splitext(input_file)[0]
        # Usa il nome del modello invece di api_provider
        model_output_name = model_name.replace('/', '-').replace('_', '-')
        out_path = f"{base}.{model_output_name}.result.jsonl"
        # evita concorrenza GPU
        n_par = max(1, int(args.n_parallel_call_per_key))
        query_chat_endpoint(
            input_file,
            out_path,
            [query_func],                   
            temp,
            n_parallel_call_per_key=n_par,
            shuffle=args.shuffle,
            model_name=model_name
        )
      return
    else:
        print("Loading MMTU from huggingface...")

        from datasets import load_dataset

        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("MMTU-benchmark/MMTU", split="train")

        # Save to JSONL file
        mmtu_file = "mmtu.jsonl"
        ds.to_json(mmtu_file, lines=True)
        
        # Usa il nome del modello invece di api_provider nel nome file
        model_output_name = args.model.replace('/', '-').replace('_', '-')
        output_file = f"mmtu.{model_output_name}.result.jsonl"            

        query_chat_endpoint(
                mmtu_file, 
                output_file,
                query_endpoints, 
                args.temperature, 
                args.n_parallel_call_per_key, 
                shuffle=args.shuffle, 
                model_name=args.model
            )

    
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", nargs="*", type=str,
                        help="input file path", default=[])
    parser.add_argument("--temperature", "-t", default=None, type=float)

    subparsers = parser.add_subparsers(dest="api_provider", required=True)

    parser_openai = subparsers.add_parser("openai", help="OpenAI API provider")
    parser_openai.add_argument("--endpoint", type=str, help="OpenAI API endpoint", required=True)
    parser_openai.add_argument("--api_key", type=str, help="OpenAI API key", required=True)
    parser_openai.add_argument("--api_version", type=str, help="OpenAI API version", required=True)
    parser_openai.add_argument("--model", type=str, help="OpenAI model name", required=True)

    parser_azure_openai = subparsers.add_parser("azure_openai", help="Azure OpenAI API provider")
    parser_azure_openai.add_argument("--endpoint", type=str, help="Azure OpenAI API endpoint", required=True)
    parser_azure_openai.add_argument("--api_key", type=str, help="Azure OpenAI API key", required=True)
    parser_azure_openai.add_argument("--api_version", type=str, help="Azure OpenAI API version", required=True)
    parser_azure_openai.add_argument("--model", type=str, help="Azure OpenAI model name", required=True)

    parser_ai_foundry = subparsers.add_parser("azure_ai_foundry", help="AI Foundry API provider")
    parser_ai_foundry.add_argument("--endpoint", type=str, help="AI Foundry API endpoint", required=True)
    parser_ai_foundry.add_argument("--api_key", type=str, help="AI Foundry API key", required=True)
    parser_ai_foundry.add_argument("--model", type=str, help="AI Foundry model name", required=True)

    parser_self_deploy = subparsers.add_parser("self_deploy", help="Self Deploy")
    parser_self_deploy.add_argument(
        "--model", 
        type=str, 
        help="Model name from registry (qwen3-8b-awq, qwen3-4b, llama-3.2-3b, llama-3.1-8b-awq) or full HuggingFace ID", 
        default="qwen3-8b-awq"
    )

    parser.add_argument("-n", "--n_parallel_call_per_key", default=1,
                        type=int, help="number of parallel calls per key")
    parser.add_argument("-l", "--log_level", default="warn", choices=["info", "warn"], type=str)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    
    # Print available models when using self_deploy
    if len(sys.argv) > 1 and sys.argv[1] == "self_deploy" and "--help" in sys.argv:
        print("\n📦 Available models in registry:")
        for model_name in MODEL_REGISTRY.keys():
            print(f"  - {model_name}")
        print("\n💡 You can also use any HuggingFace model ID directly.")
    
    args = parser.parse_args()

    main(args)