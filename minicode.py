#!/usr/bin/env python3
"""minicode - a small coding agent in one file. Python standard library only."""

import json, os, platform, shutil, signal, subprocess, sys, tempfile, time
import http.client, urllib.error, urllib.request
from datetime import date


def load_env():
    """Read KEY=value lines from a .env file next to this script. Variables already set in the shell win."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                key, found, value = line.strip().removeprefix("export ").partition("=")
                if found and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()

PROVIDER = "openrouter"
API_URL = "https://openrouter.ai/api/v1/messages"
KEY_NAME = "OPENROUTER_API_KEY"
MODEL = os.environ.get("OPENROUTER_MODEL") or "openrouter/free"
EFFORT = os.environ.get("EFFORT", "medium")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS") or 16_000)
MAX_CONTEXT = int(os.environ.get("MAX_CONTEXT") or 250_000)
COMPACT_AT = 0.8
MAX_OUTPUT = 30_000
BASH_TIMEOUT = 600  # seconds
SESSION = os.urandom(8).hex()  # one id per run, sent with every request

RESET, BOLD, DIM, BLUE, CYAN, GREEN, RED = "\033[0m", "\033[1m", "\033[2m", "\033[34m", "\033[36m", "\033[32m", "\033[31m"
if not sys.stdout.isatty():
    RESET = BOLD = DIM = BLUE = CYAN = GREEN = RED = ""


# --- Tools ---


def read(path, offset=1, limit=2000):
    with open(path, errors="replace") as f:
        lines = f.readlines()
    start = max(offset, 1) - 1
    chunk = lines[start : start + limit]
    out = "".join(f"{start + i + 1:5}| {line}" for i, line in enumerate(chunk))
    if start + limit < len(lines):
        out += f"\n[lines {start + 1}-{start + len(chunk)} of {len(lines)}; pass offset to read more]"
    return out or f"(no lines to show; the file has {len(lines)} lines)"


def edit(path, old, new, all=False):
    if not old:
        return "error: old must not be empty; use write to create a file"
    with open(path, newline="") as f:
        text = f.read()
    count = text.count(old)
    if count == 0:
        return "error: old not found in file; read the file and copy the text exactly"
    if count > 1 and not all:
        return f"error: old appears {count} times; add surrounding lines to make it unique, or pass all=true"
    with open(path, "w", newline="") as f:
        f.write(text.replace(old, new))
    return f"ok ({count} replacement{'s' if count > 1 else ''})"


def write(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"ok ({len(content.splitlines())} lines)"


def bash(cmd, timeout=BASH_TIMEOUT):
    fd, pwd_file = tempfile.mkstemp()
    os.close(fd)
    # Output goes to a file, not a pipe: a pipe stays open while any background child lives,
    # so `server &` would block until the timeout. With a file we only wait for bash itself.
    with tempfile.TemporaryFile() as out_file:
        # The trap records the final directory, so a `cd` carries over to the next call.
        proc = subprocess.Popen(
            ["bash", "-c", "trap 'pwd > \"$MINICODE_PWD\"' EXIT\n" + cmd],
            stdin=subprocess.DEVNULL, stdout=out_file, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "MINICODE_PWD": pwd_file},
        )
        try:
            proc.wait(timeout=timeout)
            note = "" if proc.returncode == 0 else f"\n[exit code {proc.returncode}]"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as err:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # the command and everything it started
            except ProcessLookupError:
                pass
            proc.wait()
            if isinstance(err, KeyboardInterrupt):
                raise
            note = f"\n[timed out after {timeout}s and was killed]"
        finally:
            with open(pwd_file) as f:
                new_dir = f.read().strip()
            os.unlink(pwd_file)
            if new_dir and os.path.isdir(new_dir):
                os.chdir(new_dir)
        out_file.seek(0)
        out = out_file.read().decode(errors="replace")
    return (out.strip() + note).strip() or "(no output)"


# name: (description, parameters, function). A trailing "?" marks an optional parameter.
TOOLS = {
    "read": (
        "Read a text file. Returns lines prefixed with line numbers. Returns at most `limit` lines "
        "(default 2000) starting at line `offset` (1-based). Fails on directories; use bash `ls` for those.",
        {"path": "string", "offset": "integer?", "limit": "integer?"},
        read,
    ),
    "edit": (
        "Replace the exact string `old` with `new` in a file. `old` must match the file exactly, including "
        "whitespace, and must appear exactly once unless all=true (which replaces every occurrence). "
        "Do not include the line-number prefixes from read. Read the file first.",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "write": (
        "Create a file, or overwrite an existing one, with `content`. Creates parent folders. "
        "Use edit to change part of an existing file.",
        {"path": "string", "content": "string"},
        write,
    ),
    "bash": (
        "Run a bash command and return stdout and stderr combined, plus the exit code when it is not 0. "
        "Use it to search (rg, grep, find, ls), run tests and builds, and use git. The working directory "
        f"persists between calls; other shell state does not. Killed after `timeout` seconds (default {BASH_TIMEOUT}). "
        f"No interactive input: stdin is closed. Output over {MAX_OUTPUT} characters is cut in the middle. "
        "To start a server or other long-running process, run it in the background with its output "
        "sent to a file (`cmd > server.log 2>&1 &`); the call then returns at once.",
        {"cmd": "string", "timeout": "integer?"},
        bash,
    ),
}
TYPES = {"string": str, "integer": int, "boolean": bool}


def make_schema():
    schema = []
    for name, (description, params, _) in TOOLS.items():
        schema.append({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {key: {"type": kind.rstrip("?")} for key, kind in params.items()},
                "required": [key for key, kind in params.items() if not kind.endswith("?")],
            },
        })
    return schema


def cap(text):
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return f"{text[:half]}\n\n[... {len(text) - MAX_OUTPUT} characters cut ...]\n\n{text[-half:]}"


def run_tool(block, bad_input):
    name, args = block["name"], block["input"]
    if block["id"] in bad_input:
        return f"error: tool input was not valid JSON: {bad_input[block['id']][:500]}"
    if name not in TOOLS:
        return f"error: unknown tool {name}"
    _, params, function = TOOLS[name]
    for key, kind in params.items():
        if key not in args and not kind.endswith("?"):
            return f"error: missing required parameter {key}"
        if key in args and not isinstance(args[key], TYPES[kind.rstrip("?")]):
            return f"error: parameter {key} must be {kind.rstrip('?')}"
    try:
        return cap(str(function(**args)))
    except Exception as err:
        return f"error: {err}"


# --- API ---


class Retryable(Exception):
    pass


def call_api(messages, system, quiet=False):
    """Send the conversation, print the reply as it streams, and return it."""
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "system": system,
        "tools": make_schema(),
        "messages": messages,
    }
    if EFFORT:
        body["output_config"] = {"effort": EFFORT}
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ.get(KEY_NAME, '')}",
        "user-agent": "minicode",
        # Keep requests in the same OpenRouter session for prompt cache reuse.
        "x-session-id": SESSION,
    }
    request = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers=headers)

    for attempt in range(5):
        try:
            return read_stream(request, quiet)
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            if (err.code not in (408, 409, 429) and err.code < 500) or attempt == 4:
                raise RuntimeError(f"API error {err.code}: {detail}") from None
            try:
                wait = min(float(err.headers.get("retry-after") or 2**attempt), 60)
            except ValueError:
                wait = 2**attempt
        except (Retryable, urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as err:
            if attempt == 4:
                raise RuntimeError(f"API connection failed: {err or type(err).__name__}") from None
            wait = 2**attempt
        print(f"{DIM}  API busy or unreachable, retrying in {wait:.0f}s...{RESET}")
        time.sleep(wait)


def add_usage(usage, new):
    """Merge token counts, skipping empty fields: some providers send null where they have no number."""
    usage.update({key: value for key, value in (new or {}).items() if value is not None})


def read_stream(request, quiet):
    blocks, bad_input, usage, stop_reason, partial, printed, complete = [], {}, {}, None, "", False, False
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:") or not line[5:].strip().startswith(b"{"):
                continue
            event = json.loads(line[5:])
            kind = event["type"]
            if kind == "message_start":
                add_usage(usage, event["message"].get("usage"))
            elif kind == "content_block_start":
                blocks.append(event["content_block"])
                partial, printed = "", False
            elif kind == "content_block_delta":
                delta, block = event["delta"], blocks[-1]
                if delta["type"] == "input_json_delta":
                    partial += delta["partial_json"]
                elif delta["type"] == "signature_delta":
                    block["signature"] = delta["signature"]
                elif delta["type"] in ("text_delta", "thinking_delta"):
                    key = "text" if delta["type"] == "text_delta" else "thinking"
                    block[key] = block.get(key, "") + delta[key]
                    if not quiet and delta[key]:
                        if not printed:
                            print(f"\n{CYAN}⏺{RESET} " if key == "text" else f"\n{DIM}✻ ", end="")
                            printed = True
                        print(delta[key], end="", flush=True)
            elif kind == "content_block_stop":
                block = blocks[-1]
                if printed:
                    print(RESET)
                if block["type"] == "tool_use" and (partial or not isinstance(block.get("input"), dict)):
                    try:
                        block["input"] = json.loads(partial or "{}")
                        if not isinstance(block["input"], dict):
                            raise ValueError
                    except ValueError:
                        block["input"], bad_input[block["id"]] = {}, partial
            elif kind == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason")
                add_usage(usage, event.get("usage"))
                complete = True
            elif kind == "message_stop":
                complete = True
            elif kind == "error":
                error = event.get("error", {})
                if error.get("type") in ("overloaded_error", "api_error", "rate_limit_error"):
                    raise Retryable(error.get("message"))
                raise RuntimeError(f"API error: {error.get('message')}")
    if not complete:  # the connection dropped before the reply was complete
        raise Retryable("the reply stream ended early")
    # The API rejects empty text blocks when they are sent back.
    blocks = [b for b in blocks if not (b["type"] == "text" and not b.get("text", "").strip())]
    context = sum(usage.get(key) or 0 for key in
                  ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"))
    return {"content": blocks, "stop_reason": stop_reason, "context": context, "bad_input": bad_input,
            "cached": usage.get("cache_read_input_tokens") or 0}


# --- Conversation ---


def with_user(messages, text):
    """Return messages plus a user text. Merges into a trailing user message so roles always alternate."""
    block = {"type": "text", "text": text}
    if messages and messages[-1]["role"] == "user":
        return messages[:-1] + [{"role": "user", "content": messages[-1]["content"] + [block]}]
    return messages + [{"role": "user", "content": [block]}]


SUMMARY_PROMPT = """Stop here and do not call any tools. The conversation is being compacted: your reply will \
replace the entire history above, so it must contain everything needed to continue the work. Write a summary with:
1. The user's goal and every instruction or preference they gave, in their words where it matters.
2. What has been done: files read, created and changed (with paths), commands run, and their results.
3. Decisions made and why, including approaches that were tried and failed.
4. The current state: what works, what is broken, what has been verified.
5. The next steps, if the task is not finished."""


def compact(state, mid_task=False):
    """Replace the whole history with a summary. No earlier turn is replayed, so nothing ties to the old transcript."""
    if not state["messages"]:
        return
    print(f"{DIM}  compacting conversation...{RESET}")
    reply = call_api(with_user(state["messages"], SUMMARY_PROMPT), state["system"], quiet=True)
    summary = "".join(b["text"] for b in reply["content"] if b["type"] == "text").strip()
    if not summary:
        print(f"{RED}  compaction failed: no summary returned{RESET}")
        return
    text = "This session continues from an earlier conversation that was compacted. Summary of it:\n\n" + summary
    if mid_task:
        text += "\n\nContinue the task from where it left off."
    state["messages"], state["context"] = with_user([], text), 0
    print(f"{DIM}  compacted{RESET}")


def over_budget(state):
    return state["context"] > MAX_CONTEXT * COMPACT_AT


def run_turn(state, text):
    state["messages"] = with_user(state["messages"], text)
    while True:
        reply = call_api(state["messages"], state["system"])
        state["context"], state["cached"] = reply["context"], reply["cached"]
        if reply["stop_reason"] == "refusal":
            print(f"{RED}⏺ The model declined this request. If it keeps happening, use /clear.{RESET}")
            return
        if not reply["content"]:
            print(f"{DIM}  (the model returned an empty reply){RESET}")
            break
        state["messages"].append({"role": "assistant", "content": reply["content"]})
        calls = [b for b in reply["content"] if b["type"] == "tool_use"]
        if not calls:
            if reply["stop_reason"] == "max_tokens":
                print(f"{DIM}  (reply was cut off at the output limit; say 'continue'){RESET}")
            break

        results = []

        def add_result(block, content):
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": content,
                            "is_error": content.startswith("error:")})

        try:
            for block in calls:
                preview = str(next(iter(block["input"].values()), "")).split("\n")[0][:70]
                print(f"\n{GREEN}⏺ {block['name']}{RESET}({DIM}{preview}{RESET})")
                if reply["stop_reason"] == "max_tokens":  # the input may be incomplete: never run it
                    result = "error: the reply was cut off before this tool call was complete. Retry with a smaller input."
                else:
                    result = run_tool(block, reply["bad_input"])
                lines = result.split("\n")
                more = f" ... +{len(lines) - 1} lines" if len(lines) > 1 else ""
                print(f"  {DIM}⎿  {lines[0][:70]}{more}{RESET}")
                add_result(block, result)
        finally:
            # Every tool call needs a result, even when the user interrupts halfway.
            for block in calls[len(results):]:
                add_result(block, "error: interrupted by the user")
            state["messages"].append({"role": "user", "content": results})

        if over_budget(state):
            compact(state, mid_task=True)
    if over_budget(state):
        compact(state)


def system_prompt():
    try:
        in_git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True).returncode == 0
    except OSError:
        in_git = False
    return f"""You are minicode, a coding agent working in the user's terminal.

Environment: {platform.system()} {platform.release()} | shell: bash | starting directory: {os.getcwd()} | \
date: {date.today()} | git repository: {"yes" if in_git else "no"}
File search: {"rg is available" if shutil.which("rg") else "rg is not installed; use find to list files and grep to search text"}.

Principles:
1. Think from first principles.
2. Do not copy what already exists. Boil the problem down to what you know is true and then reason up from there.
3. Simplify the process and question every requirement.
4. Delete any part or process you can.
5. The only rules are the laws of physics. Everything else is a recommendation.

How to work:
- Answer greetings and simple conversational requests directly without calling tools.
- Use tools only when needed for the user's task. When asked to create a file, use write to create it;
  describing a plan does not complete the task.
- Understand the actual problem before solving it. If a requirement seems wrong, say so.
- Prefer the simplest solution that works. Deleting code is better than adding it.
- Reuse what exists in the codebase and follow its conventions.
- Read a file before editing it. Use edit to change files and write to create them.
- For a task that is not trivial: explore first, state a short plan, implement it, then verify it by running \
the tests or the code. For a small task, just do it.
- Never claim something works unless you ran it. If you could not verify it, say so.
- When something fails, read the error and fix the cause. Do not retry blindly.

Safety:
- The user's instructions are rules. Stay within what they asked for.
- Do not run destructive commands (rm -rf, git reset --hard, force push), commit or push unless asked.
- Your tool calls run immediately, without the user approving them first. Be careful with anything that cannot be undone.

Communication:
- Be concise. When done, say what you changed and whether you verified it."""


def main():
    one_shot = " ".join(sys.argv[1:])
    if not os.environ.get(KEY_NAME):
        sys.exit(f"No API key for {PROVIDER}. Put {KEY_NAME}=your-key in a .env file next to minicode.py, or export it.")
    state = {"messages": [], "context": 0, "system": system_prompt()}

    if one_shot:
        try:
            run_turn(state, one_shot)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as err:
            sys.exit(f"{RED}{type(err).__name__}: {err}{RESET}")
        return

    try:
        import readline  # arrow keys and history at the prompt
    except ImportError:
        pass
    print(f"{BOLD}minicode{RESET} {DIM}| {PROVIDER} | {MODEL} | {os.getcwd()} | /clear /compact /quit{RESET}")
    while True:
        try:
            text = input("\n❯ ").strip()  # no color codes here: they confuse readline's cursor position
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not text:
            continue
        try:
            if text in ("/quit", "/exit", "/q"):
                break
            elif text == "/clear":
                state["messages"], state["context"] = [], 0
                print(f"{DIM}  cleared{RESET}")
            elif text == "/compact":
                compact(state)
            else:
                run_turn(state, text)
                print(f"{DIM}  context: {state['context']:,} / {MAX_CONTEXT:,} tokens "
                      f"({state.get('cached', 0):,} read from cache){RESET}")
        except KeyboardInterrupt:
            print(f"\n{DIM}  interrupted{RESET}")
        except Exception as err:  # never lose the session to one failed turn
            name = "" if isinstance(err, RuntimeError) else f"{type(err).__name__}: "
            print(f"{RED}⏺ {name}{err}{RESET}\n{DIM}  Your message is kept; say 'continue' to retry.{RESET}")


if __name__ == "__main__":
    main()
