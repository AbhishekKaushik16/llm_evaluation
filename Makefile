VENV          ?= $(HOME)/.venv-vllm-metal
PIP           := $(VENV)/bin/pip
PYTHON        := $(VENV)/bin/python
MODEL         ?= Qwen/Qwen2.5-3B-Instruct
HOST          ?= 0.0.0.0
PORT          ?= 8000
BASE_URL      ?= http://localhost:$(PORT)
# Prefix caching is ON by default (it helps perf in Part C). The vllm-metal
# build still cannot handle logprobs/echo even with prefix caching disabled
# (we tested), so this flag is left exposed for completeness but doesn't
# affect the eval workaround in eval_runner/gen_mc_eval.py.
PREFIX_CACHING ?= on
# Inference backend. `metal` (default) is fast but has broken logprobs in
# vllm-metal 0.2.0; `cpu` is slow but logprobs work, which is required for
# the canonical loglikelihood-based eval path.
BACKEND       ?= metal
# Sample cap per task. Empty = no cap (full benchmark).
LIMIT         ?=
LIMIT_FLAG    := $(if $(LIMIT),--limit $(LIMIT),)
# Task list override. NB: `mmlu` is a meta-task that expands to 57 subtopics —
# avoid it on the CPU backend unless you really want to spend an hour. For a
# quick canonical loglikelihood smoke test, prefer `TASKS=custom_commonsense`.
TASKS         ?= mmlu hellaswag custom_commonsense

.PHONY: install serve serve-cpu demo eval eval-gen eval-custom perf guardrails improve all

install:
	@echo "--- Installing vllm-metal (Apple Silicon) ---"
	curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
	@echo "--- Installing Python dependencies ---"
	$(PIP) install -r requirements.txt

serve:
	$(PYTHON) serve/serve.py --model $(MODEL) --host $(HOST) --port $(PORT) \
		--backend $(BACKEND) --prefix-caching $(PREFIX_CACHING)

# Shortcut for the CPU server profile (upstream vLLM, working logprobs,
# slow). Use this terminal *or* the metal one, not both — they share PORT.
serve-cpu:
	$(MAKE) serve BACKEND=cpu

demo:
	$(PYTHON) serve/demo.py --base-url $(BASE_URL) --model $(MODEL)

eval:
	$(PYTHON) eval_runner/run_eval.py \
		--base-url $(BASE_URL)/v1/completions \
		--model $(MODEL) \
		--tasks $(TASKS) \
		$(LIMIT_FLAG) \
		--output-dir eval_runner/results

# Default task list for the generation-based MC eval. Override with
# `TASKS_GEN=...` (or `TASKS=...` if you want both eval and eval-gen to
# share). HellaSwag is excluded — its full-sentence endings don't fit a
# clean letter-answer reformulation.
TASKS_GEN     ?= mmlu custom_commonsense

# Generation-based MC eval. Use this when the inference server's logprobs
# are broken (e.g. vllm-metal 0.2.0). Bypasses lm-eval's loglikelihood
# path entirely.
eval-gen:
	$(PYTHON) eval_runner/run_eval.py \
		--base-url $(BASE_URL)/v1/completions \
		--model $(MODEL) \
		--tasks $(TASKS_GEN) \
		--strategy generation \
		$(LIMIT_FLAG) \
		--output-dir eval_runner/results

eval-custom:
	$(MAKE) eval-gen TASKS_GEN=custom_commonsense

perf:
	$(PYTHON) perf/load_test.py --base-url $(BASE_URL) --model $(MODEL) --output perf/metrics.csv

guardrails:
	$(PYTHON) guardrails/validate.py --base-url $(BASE_URL) --model $(MODEL)

improve-prepare:
	$(PYTHON) improve/prepare_data.py

improve-optimize:
	$(PYTHON) improve/optimize_prompt.py --base-url $(BASE_URL) --model $(MODEL)

improve-infer:
	$(PYTHON) improve/infer.py --base-url $(BASE_URL) --model $(MODEL)

improve:
	LIMIT=$(LIMIT) bash improve/eval.sh $(BASE_URL) $(MODEL)

all: eval perf guardrails improve
	@echo "--- All tasks complete ---"
