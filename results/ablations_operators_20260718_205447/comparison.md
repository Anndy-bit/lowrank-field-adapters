# S³ Ablations (RQ4)

Same data + seed across all variants. Lower Δppl = better.

| ablation | trainable | base ppl | after ppl | Δ ppl | final loss | min loss | VRAM MB | tok/s | wall min | grad svmo | grad nmf | grad stb | grad lora |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 3,896,452 | 9.5199 | 4.4861 | -5.0338 | 1.0885 | 0.894 | 1980.4 | 6.31 | 26.85 | 0.14892 | 0.96274 | 0.26578 | 0.0 |
| no_stb | 3,437,700 | 9.5208 | 5.0744 | -4.4464 | 1.1155 | 0.9932 | 2028.3 | 6.06 | 27.96 | 0.11911 | 0.95499 | 0.0 | 0.0 |
| no_nmf | 684,740 | 9.5207 | 5.0437 | -4.477 | 1.1577 | 1.0033 | 1864.3 | 6.43 | 26.35 | 0.27976 | 0.0 | 0.50582 | 0.0 |
| svmo_only | 225,988 | 9.5208 | 9.342 | -0.1788 | 1.3424 | 1.3424 | 1927.8 | 6.33 | 26.75 | 0.3653 | 0.0 | 0.0 | 0.0 |
| nmf_only | 3,211,712 | 9.5221 | 5.0451 | -4.477 | 1.1118 | 0.9876 | 1603.9 | 6.21 | 27.3 | 0.0 | 1.03642 | 0.0 | 0.0 |

## Operator contribution (Δppl vs `full`, more negative = operator helps)

- removing/keeping-only **no_stb**: +0.5874 ppl vs full
- removing/keeping-only **no_nmf**: +0.5568 ppl vs full
- removing/keeping-only **svmo_only**: +4.855 ppl vs full
- removing/keeping-only **nmf_only**: +0.5568 ppl vs full