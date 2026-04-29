# Final Summary — Best Improvement and Lessons Learned

**Full ARC test (*n*=1,172):** baseline **81.74%**, FAISS few-shot **81.57%**, McNemar *p*=0.91 — **no significant lift**; +2.5 pp target **not** met.

**Same 500 questions:** baseline **82.40%**, few-shot greedy **83.40%**, few-shot + self-consistency (*k*=3) **83.40%** — greedy and SC tied at **417/500**; SC did not improve the headline aggregate.

The small-*n*=100 slice had looked favorable for few-shot; **full-test and merged paired analyses supersede that.**

1. **Infrastructure vs prompts** — When the server stack breaks logprobs, ship a parallel eval path or you cannot score canonical benchmarks.
2. **Slice ≠ benchmark** — Prefix slices mislead; run the full test when it fits.
3. **Long jobs** — Plug in power; long inference runs fail noisily if the laptop sleeps mid-request.

See [`README.md`](README.md) and [`improve/report.md`](improve/report.md).
