import torch
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = r"C:\Users\51615\.cache\modelscope\Ornith-1___0-9B"
REPORT_PATH = Path(__file__).parent / "v20_ornith_load_report.json"

def main():
    print("=" * 60)
    print("Ornith-1.0-9B Load Test (4-bit Quantization)")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\n[1/3] Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )
        print(f"  Tokenizer loaded. Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print(f"\n[2/3] Loading model with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print(f"  Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print(f"\n[3/3] Inspecting model structure...")
    report = {
        "model_type": type(model).__name__,
        "device": str(next(model.parameters()).device),
    }

    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            num_layers = len(inner.layers)
            report["num_layers"] = num_layers
            print(f"  Num layers: {num_layers}")

            layer_types = []
            for i, layer in enumerate(inner.layers):
                lt = type(layer).__name__
                layer_types.append(lt)
                if i < 3 or i == num_layers - 1:
                    print(f"  Layer {i}: {lt}")
                elif i == 3:
                    print(f"  ... (showing first 3 and last)")
            report["layer_types_sample"] = layer_types[:4] + layer_types[-1:]

        if hasattr(inner, "norm"):
            print(f"  Final norm: {type(inner.norm).__name__}")

    if hasattr(model, "lm_head"):
        print(f"  lm_head: {type(model.lm_head).__name__}")

    vram_mb = torch.cuda.memory_allocated() / 1024**2
    vram_reserved_mb = torch.cuda.memory_reserved() / 1024**2
    report["vram_allocated_mb"] = round(vram_mb, 1)
    report["vram_reserved_mb"] = round(vram_reserved_mb, 1)
    print(f"\n  VRAM allocated: {vram_mb:.1f} MB")
    print(f"  VRAM reserved: {vram_reserved_mb:.1f} MB")

    print(f"\n[4/4] Quick generation test (text-only)...")
    try:
        messages = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        report["generation_sample"] = decoded
        print(f"  Generated: {decoded[:100]}")
    except Exception as e:
        report["generation_error"] = str(e)
        print(f"  Generation FAILED: {e}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {REPORT_PATH}")
    print("LOAD TEST COMPLETE")

if __name__ == "__main__":
    main()