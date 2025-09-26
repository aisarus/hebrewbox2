# infer_lora_gpt2.py
import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", default="[USR] שלום, מה שלומך?\n[ASST]")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args=ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base, args.adapter)
    if torch.cuda.is_available(): model = model.to("cuda")

    gen = pipeline("text-generation", model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    out = gen(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, do_sample=True)[0]["generated_text"]
    print("\n=== OUTPUT ===\n"); print(out)

if __name__=="__main__":
    main()
