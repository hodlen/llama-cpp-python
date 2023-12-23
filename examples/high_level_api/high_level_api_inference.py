import json
import argparse

from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="../models/7B/ggml-models.bin")
parser.add_argument("-i", "--path_idx", type=str)
parser.add_argument("-ngl", "--n_gpu_layers", type=int)

args = parser.parse_args()

llm = Llama(model_path=args.model, n_gpu_layers=args.n_gpu_layers, path_idx=args.path_idx, n_ctx=128, n_batch=1)

output = llm(
    "Question: What are the names of the planets in the solar system? Answer: ",
    max_tokens=128,
    stop=["Q:", "\n"],
    echo=True,
)

print(json.dumps(output, indent=2))
