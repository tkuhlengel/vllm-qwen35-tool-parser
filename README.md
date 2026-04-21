# vllm-qwen35-tool-parser

A vLLM tool call parser plugin for **Qwen3.5** models served via vLLM. 
Implemented for running on Jetson Thor (sm_110a / CUDA 13.0), but should work
for any vLLM. Originally set up for vLLM `v0.19.0`. 

Will fix the broken tool call behavior of any qwen3.5 in vLLM by replacing
the tool parser.

## Background

vLLM's built-in `qwen3_coder` parser targets the Qwen3-Coder XML tool-call
format. The `Qwen3.5-122B-A10B-NVFP4` model's tool-call behaviour diverges
slightly from the base Qwen3-Coder parser in streaming edge cases — this
package provides a patched subclass (`Qwen35CoderToolParser`) that fixes those
issues while inheriting all upstream logic.

The package uses vLLM's `ToolParserManager` registration mechanism so the
parser can be referenced by name (`qwen35coder`) rather than by file path.

## Installation

Install into the vLLM virtual environment (editable mode recommended so local
edits take effect immediately):

```bash
source ~/.vllm/bin/activate
uv pip install --no-deps -e ~/git/vllm-qwen35-tool-parser/
```

Or as a regular install:

```bash
source ~/.vllm/bin/activate
uv pip install --no-deps ~/git/vllm-qwen35-tool-parser/
```

Verify the entry point is visible:

```bash
python3 -c "
import importlib.metadata
eps = list(importlib.metadata.entry_points(group='vllm.tool_parsers'))
print(eps)
"
```

Verify vLLM loads and registers it:

```bash
python3 -c "
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
import importlib
importlib.import_module('vllm_qwen35_tool_parser')
print('qwen35coder registered:', 'qwen35coder' in ToolParserManager.list_registered())
"
```

## Usage in vLLM serve

Add to your serve script:

```bash
vllm serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --enable-auto-tool-choice \
    --tool-parser-plugin /path/to/vllm_qwen35_tool_parser/parser.py \
    --tool-call-parser qwen35coder \
    ...
```

The key flags:
- `--tool-parser-plugin /path/to/vllm_qwen35_tool_parser/parser.py` — tells vLLM to import the
  package, which triggers the `@ToolParserManager.register_module("qwen35coder")`
  decorator and makes the name available.
- `--tool-call-parser qwen35coder` — selects the registered parser by name.

**Important:** `--tool-call-parser` takes a **name**, not a file path. The old
pattern of passing a file path there (e.g. `--tool-call-parser /path/to/file.py`)
silently falls back to no parser. Always use `--tool-parser-plugin` for the
path/module and `--tool-call-parser` for the registered name.

## What the parser does

`Qwen35CoderToolParser` extends `Qwen3CoderToolParser` (the upstream vLLM
parser for Qwen3-Coder XML tool calls) with the following fixes:

- Correct handling of streaming EOS tokens after all tool calls complete
- Proper between-tool whitespace skipping
- `prev_tool_call_arr` population on first tool-call header detection (ensures
  `finish_reason=tool_calls` is set even when argument parsing is incomplete)
- Type-aware parameter conversion via `_convert_param_value`

## Rebuilding after a vLLM venv rebuild

Since this package depends on `vllm` internals, it must be reinstalled any
time the vLLM venv is rebuilt:

```bash
source ~/.vllm/bin/activate
uv pip install --no-deps -e ~/git/vllm-qwen35-tool-parser/
```

The editable install means the source in `~/git/vllm-qwen35-tool-parser/` is
used directly — no reinstall needed for code changes, only for venv rebuilds.

## Project structure

```
vllm-qwen35-tool-parser/
├── pyproject.toml                      # package metadata + entry points
└── vllm_qwen35_tool_parser/
    ├── __init__.py                     # re-exports Qwen35CoderToolParser
    └── parser.py                       # the parser class + @register_module
```

## Environment

- Host: &lt;hostname&gt; (Jetson Thor, sm_110a)
- CUDA: 13.0
- vLLM: 0.19.1.dev2 (custom build from ~/git/vllm branch qwen35-0.19.0)
- Python: 3.13
- venv: ~/.vllm

## Known limitation: --tool-parser-plugin requires a file path

Despite the package being pip-installable, `--tool-parser-plugin` in this
version of vLLM uses `import_from_path` internally which requires a literal
`.py` file path — it cannot resolve an installed module name.

Use the absolute path to `parser.py`:

```bash
--tool-parser-plugin /path/to/vllm-qwen35-tool-parser/vllm_qwen35_tool_parser/parser.py
--tool-call-parser qwen35coder
```

The entry point in `pyproject.toml` is kept for future vLLM versions that
may support module-name resolution in `--tool-parser-plugin`.
