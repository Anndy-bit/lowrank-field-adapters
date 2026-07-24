# S³ Ablations (RQ4)

Same data + seed across all variants. Lower Δppl = better.

| ablation | trainable | base ppl | after ppl | Δ ppl | final loss | min loss | VRAM MB | tok/s | wall min | grad svmo | grad nmf | grad stb | grad lora |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nmf_b3 | 1,204,392 | 9.5197 | 5.3651 | -4.1546 | 1.149 | 1.0352 | 1206.5 | 5.93 | 28.58 | 0.0 | 0.81905 | 0.0 | 0.0 |
| no_nmf_h64 | 1,311,940 | 9.5207 | 5.1374 | -4.3833 | 1.1665 | 1.0166 | 1825.3 | 5.36 | 31.6 | 0.29372 | 0.0 | 0.50921 | 0.0 |