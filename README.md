# Apple MLX GPT (M3 Air)

From-scratch inspectable GPT on MacBook Air M3 (8 GB unified memory). Host-side
CLI / tokenizer / NumPy reference come from [llm-gpu-8](https://github.com/dtelcore/llm-gpu-8);
the device layer is MLX ops + explicit VJPs (no autograd).

Process budget: **2 GB** of unified memory. Soft machine guard: **5.5 GB**.

```bash
# Python 3.11 or 3.12
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python setup/2_test_workspace.py          # Metal + matmul + memory APIs
python -m tests.parity.run_parity
python auto_train.py --steps 20 --prompt "once upon a" --no-prompt
```

Default recipe: `setup/story_sub1m_config.json` (C=128, L=4, T=128, batch 8).
Keep the lid open on this fanless Air; long GEMMs will thermal-throttle.

Live path: `model/mlx/ops.py` (forward primitives + hand VJPs). Do not use
`mx.value_and_grad`, `mlx.nn`, or `mlx.optimizers` on the train loop.
