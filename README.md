# minicode

Two terminal coding agents, inspired by [nanocode](https://github.com/1rgs/nanocode).
Python 3.9+, macOS/Linux and bash. No Python packages to install.

| Harness | Features |
|---|---|
| `minicode.py` | Minimal agent with read, write, edit, bash, streaming and automatic compaction. |
| `agent.py` | Adds parallel workers, isolated editing, web search/fetch, plans and execution limits. |

## Setup and usage

Create `.env` next to the scripts:

```env
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=openrouter/free
```

The model defaults to `openrouter/free`. Change `OPENROUTER_MODEL` to use another
OpenRouter model with tool calling. Optionally set `SUBAGENT_MODEL` for workers;
otherwise they inherit the coordinator's model. Shell variables override `.env`.

```bash
python3 minicode.py                         # interactive, current directory
python3 agent.py                            # interactive, current directory
python3 agent.py "fix the failing test"     # one task, then exit
python3 agent.py --cwd /path/to/project      # choose a project
```

Both scripts also work when invoked by absolute path from another project.

## Parallel workers

`agent.py` has a coordinator (`[main]`) and implementation, research and review
workers. The coordinator can work while workers run, collect their results and
integrate completed changes. Workers stop after returning a result; extra tasks
queue when all slots are occupied. Workers cannot spawn other workers.

Implementers edit and run tests in temporary copies of the project. Your repository
is not touched, and git commands do not work inside a copy. The coordinator reviews diffs and
integrates changes explicitly. Conflicting or out-of-scope changes are refused.
Research and review workers have read/search tools only.

Worker snapshots exclude secrets files, dependencies, caches and symlinks.
Workspaces and results remain in the temporary directory printed at startup.
There is no persistent memory or session resumption.

## Web tools and output

- `web_search(query)`: keyless DuckDuckGo HTML search, using only the standard library.
- `web_fetch(url)`: extracts text from a supplied public URL. No browser rendering or PDF support.
- Searches are serialized across agents and cached per agent. Blocking, network failures
  and page-format changes produce errors; this is not a guaranteed search API.
- Reasoning returned by the provider streams in dim gray with `✻` and an agent label.
  `agent.py` buffers final answer text; `minicode.py` streams answer text too.

## Configuration

Edit `AgentConfig` near the top of `agent.py` for runtime settings:

| Field | Default |
|---|---|
| `max_parallel` | 5 active workers, excluding the coordinator |
| `max_workers` | 15 workers per user task |
| `worker_steps` | 30 model request attempts per worker |
| `worker_timeout` | 300 seconds from startup; queue time excluded |
| `effort` | `"medium"` |
| `show_reasoning` | `True` |

Other limits are defined there too. Retries and
compaction count toward request limits. Provider rate limits still apply.

For `minicode.py`, `.env` also supports `EFFORT` (default `medium`), `MAX_TOKENS`
(`16000`) and `MAX_CONTEXT` (`250000`). Adjust context budgets to your model.

## Commands

Both: `/clear`, `/compact`, `/quit` or `/exit`.

`agent.py` also has `/agents`, `/plan` and `/cancel AGENT_ID` at the prompt.
Ctrl-C during a task interrupts execution; `agent.py` also cancels its workers
and retains partial work.

## Execution and tests

Tools run immediately. Shell commands have your permissions; worker workspaces
are not security sandboxes. Use a container or a repository with your work backed up.

```bash
python3 -S -m unittest -v test_agent.py
```
