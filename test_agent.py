"""Offline behavioral tests. Run: python3 -m unittest -v test_agent.py."""

import io
import fcntl
from email.message import Message
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import patch

import agent


class SilentLogger:
    def emit(self, *args, **kwargs):
        pass


def reply(text="", calls=None, reason=None):
    return {"content": ([{"type": "text", "text": text}] if text else []) + (calls or []),
            "stop_reason": reason or ("tool_use" if calls else "end_turn"),
            "invalid": set(), "usage": {"input_tokens": 100, "output_tokens": 10},
            "context": 110, "cached": None}


def tool(name, args, identity="call-1"):
    return {"type": "tool_use", "name": name, "id": identity, "input": args}


def fake_worker(record, config, limiter, events):
    """Actual spawned processes, with deterministic work instead of a remote model."""
    os.setsid()
    options = json.loads(record["task"])
    if options.get("shell"):
        process = subprocess.Popen(["bash", "-c", "sleep 60"], cwd=record["workspace"])
        Path(record["folder"], "child-pid").write_text(str(process.pid))
    if options.get("abrupt"):
        os._exit(7)
    if options.get("path"):
        Path(record["workspace"], options["path"]).write_text(options.get("content", "worker"))
    if options.get("reasoning"):
        events.put({"agent": record["id"], "event": "reasoning_delta", "detail": "Live worker thought"})
    time.sleep(options.get("sleep", 0))
    if options.get("reasoning"):
        events.put({"agent": record["id"], "event": "reasoning_end", "detail": ""})
    result = {"status": "failed" if options.get("fail") else "completed", "summary": "worker result"}
    Path(record["folder"], "worker-result.json").write_text(json.dumps(result))


def scripted_worker(record, config, limiter, events):
    """Exercise the real worker loop, file tools and bash in a spawned process."""
    index = 0
    relative = record["owned_paths"][0]

    def call(client, messages, system, tools, quiet=False):
        nonlocal index
        client.charge()
        index += 1
        if index == 1:
            return reply(calls=[tool("write", {"path": relative, "content": "print('worker-ok')\n"})])
        if index == 2:
            return reply(calls=[tool("bash", {"cmd": f"{shutil.which('python3')} {relative}"})])
        return reply("Implemented " + relative + "; verified by running Python, output worker-ok.")

    with patch.object(agent.ModelClient, "call", call):
        agent.worker_main(record, config, limiter, events)


def pid_active(pid):
    try:
        result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
        return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")
    except OSError:
        return False


def hold_rate_lock(path, ready):
    os.setsid()
    with open(path, "r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        Path(ready).write_text("ready")
        time.sleep(60)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agent-test-")
        self.folder = Path(self.temp.name)
        self.root, self.session = self.folder / "project", self.folder / "session"
        self.root.mkdir()
        self.session.mkdir()
        self.config = agent.AgentConfig(requests_per_minute=60000, tool_timeout=3)
        self.ctx = mp.get_context("spawn")
        self.limiter = agent.RateLimiter(60000, self.session / "rate-limit.bin")
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.close()
        self.temp.cleanup()

    def manager(self, target=fake_worker):
        manager = agent.WorkerManager(self.config, self.root, self.session, SilentLogger(),
                                      self.limiter, self.ctx, worker_target=target)
        self.managers.append(manager)
        return manager

    def runner(self, role="coordinator", **kwargs):
        return agent.AgentRunner(self.config, self.root, self.limiter, lambda *args: None,
                                 role=role, **kwargs)

    def wait_for(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.03)
        self.fail("condition did not become true")

    def test_six_jobs_respect_five_slots_and_results_deliver_once(self):
        manager = self.manager()
        ids = [manager.spawn(json.dumps({"sleep": 1.5}), role="researcher")["id"] for _ in range(6)]
        self.wait_for(lambda: sum(r["status"] == "running" for r in manager.status()) == 5)
        self.assertEqual(sum(r["status"] == "queued" for r in manager.status()), 1)
        while any(r["status"] not in agent.TERMINAL for r in manager.status()):
            self.assertLessEqual(sum(r["status"] == "running" for r in manager.status()), 5)
            time.sleep(.03)
        self.assertEqual({r["id"] for r in manager.collect()}, set(ids))
        self.assertEqual(manager.collect(), [])

    def test_timeout_kills_worker_and_shell_and_preserves_partial_files(self):
        self.config.worker_timeout = .8
        manager = self.manager()
        record = manager.spawn(json.dumps({"sleep": 60, "shell": True, "path": "partial.txt"}),
                               owned_paths=["partial.txt"])
        pid_path = self.session / record["id"] / "child-pid"
        self.wait_for(pid_path.exists)
        pid = int(pid_path.read_text())
        self.wait_for(lambda: manager.status(record["id"])["status"] == "timed_out")
        self.wait_for(lambda: not pid_active(pid))
        self.assertTrue(Path(record["workspace"], "partial.txt").exists())
        self.assertFalse((self.root / "partial.txt").exists())

    def test_cancel_running_queued_and_immediate_startup(self):
        self.config.max_parallel = 1
        manager = self.manager()
        first = manager.spawn(json.dumps({"sleep": 60, "shell": True}), role="researcher")
        second = manager.spawn("{}", role="researcher")
        pid_path = self.session / first["id"] / "child-pid"
        self.wait_for(pid_path.exists)
        self.assertEqual(manager.cancel(second["id"])["status"], "cancelled")
        pid = int(pid_path.read_text())
        manager.cancel(first["id"])
        self.wait_for(lambda: not pid_active(pid))
        third = manager.spawn(json.dumps({"sleep": 60}), role="researcher")
        self.wait_for(lambda: manager.status(third["id"])["status"] == "running")
        manager.cancel(third["id"])
        self.assertEqual(manager.status(third["id"])["status"], "cancelled")
        self.assertFalse(manager.processes)

    def test_crashed_worker_does_not_break_queue(self):
        self.config.max_parallel = 1
        manager = self.manager()
        bad = manager.spawn('{"abrupt": true}', role="researcher")
        good = manager.spawn("{}", role="researcher")
        self.wait_for(lambda: manager.status(good["id"])["status"] == "completed")
        self.assertEqual(manager.status(bad["id"])["status"], "failed")

    def test_worker_count_and_role_restrictions(self):
        self.config.max_workers = 1
        manager = self.manager()
        with self.assertRaises(ValueError):
            manager.spawn("task", owned_paths=[])
        with self.assertRaises(ValueError):
            manager.spawn("task", owned_paths=["../outside"])
        manager.spawn("{}", role="researcher")
        with self.assertRaisesRegex(ValueError, "max_workers"):
            manager.spawn("{}", role="reviewer")
        for role in ("researcher", "reviewer"):
            runner = self.runner(role)
            self.assertNotIn("bash", runner.tools)
            self.assertNotIn("write", runner.tools)
            self.assertNotIn("spawn_agent", runner.tools)
            self.assertTrue(runner.tool(tool("write", {"path": "bad", "content": "bad"}), set()).startswith("error:"))

    def test_parallel_real_workers_and_coordinator_integrate_and_run(self):
        manager = self.manager(scripted_worker)
        first = manager.spawn("implement first", owned_paths=["first.py"])
        second = manager.spawn("implement second", owned_paths=["second.py"])
        (self.root / "coordinator.py").write_text("print('main-ok')\n")
        self.wait_for(lambda: all(r["status"] in agent.TERMINAL for r in manager.status()))
        self.assertTrue(all(r["status"] == "completed" for r in manager.status()), manager.status())
        self.assertFalse((self.root / "first.py").exists())
        self.assertTrue(manager.integrate(first["id"])["integrated"])
        self.assertTrue(manager.integrate(second["id"])["integrated"])
        command = " && ".join(f"'{shutil.which('python3')}' {name}.py" for name in ("first", "second", "coordinator"))
        result = self.runner().bash(command)
        self.assertIn("exit code 0", result)
        self.assertEqual(result.count("worker-ok"), 2)
        self.assertIn("main-ok", result)
        self.assertEqual(manager.integrate(first["id"])["applied"], [])

    def test_conflict_refuses_entire_change_set(self):
        (self.root / "shared.txt").write_text("original")
        workspaces = agent.WorkspaceManager(self.root, self.session, self.config.workspace_bytes)
        record = {**workspaces.create("worker"), "status": "completed", "owned_paths": ["."]}
        Path(record["workspace"], "shared.txt").write_text("worker")
        Path(record["workspace"], "new.txt").write_text("new")
        (self.root / "shared.txt").write_text("coordinator")
        result = workspaces.integrate(record)
        self.assertFalse(result["integrated"])
        self.assertEqual((self.root / "shared.txt").read_text(), "coordinator")
        self.assertFalse((self.root / "new.txt").exists())

    def test_out_of_scope_shell_changes_are_not_integrated(self):
        manager = self.manager()
        record = manager.spawn('{"path": "unowned.txt"}', owned_paths=["owned.txt"])
        self.wait_for(lambda: manager.status(record["id"])["status"] == "completed")
        self.assertFalse(manager.integrate(record["id"])["integrated"])
        self.assertFalse((self.root / "unowned.txt").exists())

    def test_snapshot_excludes_secrets_links_and_preserves_dirty_git_state(self):
        if not shutil.which("git"):
            self.skipTest("git not installed")
        def git(*args):
            return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)
        git("init")
        (self.root / "tracked.txt").write_text("committed")
        git("add", "tracked.txt")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
        (self.root / "tracked.txt").write_text("dirty-user-change")
        (self.root / "untracked.txt").write_text("untracked-user-change")
        (self.root / ".env").write_text("SECRET=never-copy")
        (self.root / "link").symlink_to(self.root / "tracked.txt")
        workspaces = agent.WorkspaceManager(self.root, self.session, self.config.workspace_bytes)
        record = {**workspaces.create("worker"), "status": "completed", "owned_paths": ["tracked.txt"]}
        work = Path(record["workspace"])
        self.assertEqual((work / "tracked.txt").read_text(), "dirty-user-change")
        self.assertEqual((work / "untracked.txt").read_text(), "untracked-user-change")
        self.assertFalse((work / ".env").exists())
        self.assertFalse((work / "link").exists())
        self.assertFalse((work / ".git").exists())
        (work / "tracked.txt").write_text("worker-change")
        self.assertTrue(workspaces.integrate(record)["integrated"])
        self.assertEqual((self.root / "tracked.txt").read_text(), "worker-change")
        self.assertEqual((self.root / "untracked.txt").read_text(), "untracked-user-change")
        # The user's repository is left alone: no worktree is registered for the worker.
        self.assertEqual(len(git("worktree", "list").stdout.splitlines()), 1)

    def test_independent_cwds_and_file_tool_boundaries(self):
        (self.root / "a").mkdir()
        (self.root / "b").mkdir()
        first, second = self.runner(), self.runner()
        original_cwd = Path.cwd()
        first.bash("cd a")
        second.bash("cd b")
        first.write("first.txt", "a")
        second.write("second.txt", "b")
        self.assertTrue((self.root / "a/first.txt").exists())
        self.assertTrue((self.root / "b/second.txt").exists())
        self.assertEqual(Path.cwd(), original_cwd)
        with self.assertRaises(ValueError):
            first.write("../../outside", "bad")
        (self.root / "a/link").symlink_to(self.root / "b/second.txt")
        with self.assertRaises(ValueError):
            first.write("link", "bad")
        implementer = self.runner("implementer", owned_paths=["allowed.py"])
        with self.assertRaises(ValueError):
            implementer.write("other.py", "bad")

    def test_simple_reply_uses_no_tools_and_budget_stops_loops(self):
        runner = self.runner()
        with patch.object(runner.client, "call", return_value=reply("OK")), patch.object(runner, "tool") as run_tool:
            self.assertEqual(runner.run("reply with OK"), "OK")
            run_tool.assert_not_called()
        runner = self.runner()
        runner.step_limit = 2
        def loop(*args, **kwargs):
            runner.charge()
            return reply(calls=[tool("list_files", {})])
        with patch.object(runner.client, "call", side_effect=loop):
            with self.assertRaises(agent.StepLimit):
                runner.run("keep working")
        self.assertEqual(runner.steps, 2)

    def test_file_creation_loop_and_tool_errors(self):
        runner = self.runner()
        calls = [reply(calls=[tool("write", {"path": "page.html", "content": "<h1>Hello</h1>"})]), reply("Created page.html")]
        with patch.object(runner.client, "call", side_effect=calls):
            self.assertEqual(runner.run("create a page"), "Created page.html")
        self.assertEqual((self.root / "page.html").read_text(), "<h1>Hello</h1>")
        self.assertTrue(runner.tool(tool("write", {"path": "bad", "content": "bad", "extra": True}), set()).startswith("error:"))
        self.assertTrue(runner.tool(tool("write", {"path": "bad", "content": "bad"}), {"call-1"}).startswith("error:"))
        self.assertFalse((self.root / "bad").exists())

    def test_stream_parser_and_incomplete_response(self):
        runner = self.runner()
        events = [
            {"type": "message_start", "message": {"model": "selected/model", "usage": {"input_tokens": 10, "cache_read_input_tokens": None}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "write", "input": {}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"path":"ok","content":"yes"}'}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        def data(items):
            return io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in items))
        with patch('urllib.request.urlopen', return_value=data(events)):
            result = runner.client.read_stream(None)
        self.assertEqual(result["content"][0]["input"]["content"], "yes")
        self.assertEqual(result["context"], 15)
        self.assertIsNone(result["cached"])
        self.assertEqual(runner.client.last_model, "selected/model")
        with patch('urllib.request.urlopen', return_value=data(events[:-1])):
            with self.assertRaises(agent.Retryable):
                runner.client.read_stream(None)
        events[2]["delta"]["partial_json"] = "{broken"
        with patch('urllib.request.urlopen', return_value=data(events)):
            self.assertEqual(runner.client.read_stream(None)["invalid"], {"t"})

    def test_reasoning_deltas_are_emitted_before_response_finishes(self):
        runner = self.runner()
        emitted = []
        runner.client.emit = lambda event, detail='': emitted.append((event, detail))

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(response):
                def data(event):
                    return b'data: ' + json.dumps(event).encode() + b'\n\n'
                yield data({'type': 'message_start', 'message': {'usage': {}}})
                yield data({'type': 'content_block_start', 'index': 0,
                            'content_block': {'type': 'thinking', 'thinking': ''}})
                yield data({'type': 'content_block_delta', 'index': 0,
                            'delta': {'type': 'thinking_delta', 'thinking': 'First '}})
                self.assertIn(('reasoning_delta', 'First '), emitted)
                yield data({'type': 'content_block_delta', 'index': 0,
                            'delta': {'type': 'thinking_delta', 'thinking': 'second.'}})
                self.assertIn(('reasoning_delta', 'second.'), emitted)
                yield data({'type': 'content_block_stop', 'index': 0})
                self.assertEqual(emitted[-1], ('reasoning_end', ''))
                yield data({'type': 'content_block_start', 'index': 1,
                            'content_block': {'type': 'text', 'text': 'Done'}})
                yield data({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})
                yield data({'type': 'message_stop'})

        with patch('urllib.request.urlopen', return_value=Response()):
            result = runner.client.read_stream(None)
        self.assertEqual(result['content'][0]['thinking'], 'First second.')
        self.assertEqual([text for event, text in emitted if event == 'reasoning_delta'], ['First ', 'second.'])

    def test_reasoning_hidden_for_quiet_requests_and_reset_on_disconnect(self):
        runner = self.runner()
        events = [
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': 'Initial '}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'partial'}},
        ]
        def stream(complete=False):
            items = events + ([{'type': 'message_stop'}] if complete else [])
            return io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in items))
        emitted = []
        runner.client.emit = lambda event, detail='': emitted.append((event, detail))
        with patch('urllib.request.urlopen', return_value=stream()):
            with self.assertRaises(agent.Retryable):
                runner.client.read_stream(None)
        self.assertEqual(emitted, [('reasoning_delta', 'Initial '), ('reasoning_delta', 'partial'), ('reasoning_end', '')])
        emitted.clear()
        with patch('urllib.request.urlopen', return_value=stream(True)):
            runner.client.read_stream(None, quiet=True)
        self.assertFalse(emitted)
        runner.config.show_reasoning = False
        with patch('urllib.request.urlopen', return_value=stream(True)):
            runner.client.read_stream(None)
        self.assertFalse(emitted)

    def test_reasoning_terminal_color_continuation_and_worker_labels(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        output = Terminal()
        with patch('sys.stdout', output):
            logger = agent.EventLogger()
            logger.emit('main', 'reasoning_delta', detail='First ')
            first = output.getvalue()
            self.assertIn('\033[2;90m', first)
            self.assertIn('✻ [main]', first)
            self.assertFalse(first.endswith('\n'))
            logger.emit('main', 'reasoning_delta', detail='second')
            self.assertEqual(output.getvalue().count('✻ [main]'), 1)
            logger.emit('researcher-1', 'reasoning_delta', detail='Worker thought')
            logger.emit('main', 'tool', detail='read')
            self.assertIsNone(logger.reasoning_agent)
            self.assertIn('✻ [researcher-1]', output.getvalue())
            self.assertTrue(output.getvalue().endswith('[main] tool: read\n'))
        output = io.StringIO()
        with patch('sys.stdout', output):
            logger = agent.EventLogger()
            logger.emit('main', 'reasoning_delta', detail='Plain')
            logger.emit('main', 'reasoning_end')
        self.assertNotIn('\033[', output.getvalue())
        self.assertEqual(output.getvalue(), '\n✻ [main] Plain\n')

    def test_worker_reasoning_is_forwarded_while_worker_is_running(self):
        manager = self.manager()
        received = []
        manager.logger.emit = lambda agent_id, event, **kwargs: received.append((agent_id, event, kwargs.get('detail', '')))
        record = manager.spawn('{"reasoning": true, "sleep": 0.8}', role='researcher')
        self.wait_for(lambda: any(event == 'reasoning_delta' for _, event, _ in received))
        self.assertEqual(manager.status(record['id'])['status'], 'running')
        self.wait_for(lambda: manager.status(record['id'])['status'] == 'completed')
        events = [event for agent_id, event, _ in received if agent_id == record['id']]
        self.assertLess(events.index('reasoning_delta'), events.index('reasoning_end'))
        self.assertLess(events.index('reasoning_end'), events.index('completed'))

    def test_permanent_api_error_is_not_retried(self):
        runner = self.runner()
        error = urllib.error.HTTPError(agent.API_URL, 401, "unauthorized", {}, io.BytesIO(b"invalid key"))
        with patch('urllib.request.urlopen', side_effect=error) as transport:
            with self.assertRaisesRegex(RuntimeError, "401"):
                runner.client.call([], "system", [])
            self.assertEqual(transport.call_count, 1)

    def test_web_extraction_search_cache_limits_and_private_urls(self):
        parser = agent.TextExtractor()
        parser.feed("<h1>Title</h1><script>secret()</script><p>Useful &amp; readable</p>")
        self.assertIn("Useful & readable", parser.text())
        self.assertNotIn("secret", parser.text())
        with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', ('127.0.0.1', 80))]):
            with self.assertRaises(ValueError):
                agent.public_url("http://localhost")
        with self.assertRaises(ValueError):
            agent.public_url("file:///etc/passwd")

    def test_keyless_html_search_normalizes_deduplicates_limits_and_caches(self):
        class Response(io.BytesIO):
            def __init__(self, body):
                super().__init__(body.encode())
                self.headers = Message()
                self.headers['Content-Type'] = 'text/html; charset=utf-8'
            def getcode(self):
                return 200
        page = """<html><body>
        <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F%3Fa%3D1%26b%3D2&amp;rut=x">Docs &amp; <b>examples</b></a></h2>
        <a class="result__snippet">A &amp; <b>B</b></a>
        <h2><a class="result__a" href="https://example.com/?a=1&amp;b=2">Duplicate</a></h2>
        <div class="result__snippet">Duplicate snippet</div>
        <a class="result__a" href="javascript:bad()">Invalid</a>
        <a class="result__snippet">Invalid snippet</a>
        <a class="result__a" href="https://other.example">Other<br> result</a>
        <div class="result__snippet">Alternative shape</div>
        </body></html>"""
        tools = agent.WebTools(self.config)
        with patch('urllib.request.urlopen', return_value=Response(page)) as transport:
            result = tools.search(' topic & more ', 2)
            self.assertEqual(result['provider'], 'duckduckgo')
            self.assertEqual(len(result['results']), 2)
            self.assertEqual(result['results'][0], {'title': 'Docs & examples', 'url': 'https://example.com/?a=1&b=2', 'snippet': 'A & B'})
            self.assertEqual(result['results'][1]['snippet'], 'Alternative shape')
            self.assertTrue(result['untrusted_source_content'])
            tools.search('topic & more', 2)
            self.assertEqual(transport.call_count, 1)
            request = transport.call_args.args[0]
            self.assertIn('q=topic+%26+more', request.full_url)
            self.assertIn('html.duckduckgo.com/html/', request.full_url)
            self.assertIsNone(request.get_header('X-subscription-token'))
        parser = agent.SearchResultsParser()
        parser.feed(page)
        self.assertEqual(len(parser.normalized(1)), 1)
        self.assertEqual(parser.result_url('/l/?ad_provider=x'), '')
        self.assertEqual(parser.result_url('https://user:pass@example.com'), '')

    def test_html_search_challenges_errors_empty_results_and_limits(self):
        class Response(io.BytesIO):
            def __init__(self, body, status=200):
                super().__init__(body)
                self.status = status
                self.headers = Message()
                self.headers['Content-Type'] = 'text/html; charset=utf-8'
            def getcode(self):
                return self.status
        runner = self.runner()
        for body, status, expected in [
            (b'<form id="challenge-form">Verify</form>', 200, 'verification challenge'),
            (b'<div class="anomaly-modal">Verify</div>', 202, 'verification challenge'),
            (b'<html>Unexpected page</html>', 200, 'page format changed'),
            (b'x' * 2_000_001, 200, 'download limit'),
        ]:
            with patch('urllib.request.urlopen', return_value=Response(body, status)):
                result = runner.tool(tool('web_search', {'query': 'topic'}), set())
                self.assertTrue(result.startswith('error:'), result)
                self.assertIn(expected, result)
            self.assertFalse(runner.web.cache)  # failed searches are never cached
        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError('network unavailable')):
            result = runner.tool(tool('web_search', {'query': 'topic'}), set())
            self.assertIn('network unavailable', result)
        with patch('urllib.request.urlopen', return_value=Response(b'<div class="no-results">No results found</div>')):
            self.assertEqual(runner.web.search('no matches')['results'], [])
        for query, limit in [('', 5), ('  ', 5), ('topic', 0), ('topic', 11)]:
            with self.assertRaises(ValueError):
                runner.web.search(query, limit)

    def test_searches_share_one_slot_and_release_it_after_failure(self):
        lock = self.session / "search.lock"
        first, second = agent.WebTools(self.config, lock), agent.WebTools(self.config, lock)
        active, peak = 0, 0
        guard = threading.Lock()
        def search(query, limit):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(.1)
            with guard:
                active -= 1
            return [{"title": "Result", "url": "https://example.com", "snippet": "Found"}]
        with patch.object(first, 'duckduckgo_search', side_effect=search), patch.object(second, 'duckduckgo_search', side_effect=search):
            threads = [threading.Thread(target=web.search, args=("topic",)) for web in (first, second)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
            self.assertEqual(peak, 1)
            self.assertTrue(first.cache)
            self.assertTrue(second.cache)
        with patch.object(first, 'duckduckgo_search', side_effect=ValueError('failed')):
            with self.assertRaisesRegex(ValueError, 'failed'):
                first.search('different')
        with patch.object(second, 'duckduckgo_search', side_effect=search):
            self.assertTrue(second.search('different')['results'])

    def test_configuration_defaults_overrides_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            config = agent.AgentConfig.from_env()
            self.assertEqual(config.max_parallel, 5)
            self.assertEqual(config.worker_timeout, 300)
            self.assertEqual(config.model, "openrouter/free")
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "custom/main", "SUBAGENT_MODEL": "custom/worker",
                                     "MAX_PARALLEL_SUBAGENTS": "2", "SUBAGENT_TIMEOUT_SECONDS": "900"}, clear=True):
            config = agent.AgentConfig.from_env()
            self.assertEqual((config.model, config.worker_model, config.max_parallel, config.worker_timeout),
                             ("custom/main", "custom/worker", 5, 300))
        config = agent.AgentConfig(max_parallel=2, worker_timeout=900)
        self.assertEqual((config.max_parallel, config.worker_timeout), (2, 900))
        for kwargs in ({"max_parallel": 0}, {"max_parallel": 1.5}, {"worker_timeout": float('nan')}):
            with self.assertRaises(ValueError):
                agent.AgentConfig(**kwargs)

    def test_coordinator_collects_integrates_and_verifies_worker_results(self):
        manager = self.manager(scripted_worker)
        runner = self.runner(manager=manager)
        verified = False

        def coordinator(messages, system, tools, quiet=False):
            nonlocal verified
            runner.charge()
            records = manager.status()
            if not records:
                return reply(calls=[tool("spawn_agent", {"task": "first", "owned_paths": ["first.py"]}, "spawn-1"),
                                    tool("spawn_agent", {"task": "second", "owned_paths": ["second.py"]}, "spawn-2")])
            if not (self.root / "main.py").exists():
                return reply(calls=[tool("write", {"path": "main.py", "content": "print('main-ok')\n"})])
            if any(r["status"] not in agent.TERMINAL for r in records):
                return reply("Waiting for workers.")
            unfinished = [r for r in records if not r["integrated"]]
            if unfinished:
                for record in records:
                    self.assertTrue(manager.records[record["id"]]["delivered"])
                return reply(calls=[tool("integrate_agent", {"agent_id": r["id"]}, r["id"]) for r in unfinished])
            if not verified:
                verified = True
                return reply(calls=[tool("bash", {"cmd": f"'{shutil.which('python3')}' first.py && '{shutil.which('python3')}' second.py && '{shutil.which('python3')}' main.py"})])
            last_result = messages[-1]["content"][0]
            self.assertIn("worker-ok", last_result["content"])
            self.assertIn("main-ok", last_result["content"])
            return reply("Implemented and verified all three files.")

        with patch.object(runner.client, "call", side_effect=coordinator):
            answer = runner.run("Implement two independent components in parallel and a main file.")
        self.assertIn("verified", answer)
        self.assertTrue(all(r["integrated"] for r in manager.status()))
        self.assertEqual(manager.collect(), [])

    def test_rate_limiter_recovers_when_lock_holder_is_killed(self):
        ready = self.folder / "ready"
        process = self.ctx.Process(target=hold_rate_lock, args=(str(self.limiter.path), str(ready)))
        process.start()
        try:
            self.wait_for(ready.exists)
            agent.stop_process(process)
            deadline = time.monotonic() + 1
            def check():
                if time.monotonic() > deadline:
                    raise TimeoutError("lock did not recover")
            self.limiter.acquire(check)
        finally:
            if process.is_alive():
                agent.stop_process(process)

    def test_integration_deletion_mode_and_rollback_on_io_failure(self):
        (self.root / "delete.txt").write_text("remove")
        (self.root / "keep.txt").write_text("original")
        workspaces = agent.WorkspaceManager(self.root, self.session, self.config.workspace_bytes)
        record = {**workspaces.create("worker"), "status": "completed", "owned_paths": ["."]}
        work = Path(record["workspace"])
        (work / "delete.txt").unlink()
        (work / "keep.txt").write_text("updated")
        (work / "keep.txt").chmod(0o755)
        original_copy = shutil.copy2
        def failing_copy(source, dest, *args, **kwargs):
            if Path(source).resolve() == (work / "keep.txt").resolve() and Path(dest).resolve() == (self.root / "keep.txt").resolve():
                raise OSError("simulated disk error")
            return original_copy(source, dest, *args, **kwargs)
        with patch('shutil.copy2', side_effect=failing_copy):
            with self.assertRaisesRegex(OSError, "disk error"):
                workspaces.integrate(record)
        self.assertEqual((self.root / "delete.txt").read_text(), "remove")
        self.assertEqual((self.root / "keep.txt").read_text(), "original")
        self.assertTrue(workspaces.integrate(record)["integrated"])
        self.assertFalse((self.root / "delete.txt").exists())
        self.assertEqual((self.root / "keep.txt").stat().st_mode & 0o777, 0o755)

    def test_fetch_html_cache_truncation_redirect_checks_and_errors(self):
        class Response(io.BytesIO):
            def __init__(self, body, kind="text/html"):
                super().__init__(body)
                self.headers = Message()
                self.headers['Content-Type'] = kind + "; charset=utf-8"
            def geturl(self):
                return "https://example.com/final"
        tools = agent.WebTools(self.config)
        with patch.object(agent, 'public_url'), patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = Response(b'<h1>Title</h1><p>Body</p><script>hidden</script>')
            result = tools.fetch('https://example.com')
            self.assertIn('Body', result['text'])
            self.assertNotIn('hidden', result['text'])
            self.assertEqual(result['url'], 'https://example.com/final')
            tools.fetch('https://example.com')
            self.assertEqual(opener.return_value.open.call_count, 1)
            opener.return_value.open.return_value = Response(b'x' * 2_000_001, 'text/plain')
            self.assertTrue(tools.fetch('https://example.com/large')['truncated'])
            opener.return_value.open.return_value = Response(b'PDF', 'application/pdf')
            with self.assertRaisesRegex(ValueError, 'unsupported content'):
                tools.fetch('https://example.com/file.pdf')
            opener.return_value.open.side_effect = urllib.error.URLError('unreachable')
            runner = self.runner()
            self.assertIn('unreachable', runner.tool(tool('web_fetch', {'url': 'https://example.com/error'}), set()))
        with patch.object(agent, 'public_url', side_effect=ValueError('private redirect')):
            with self.assertRaisesRegex(ValueError, 'private redirect'):
                agent.PublicRedirect().redirect_request(None, None, 302, '', {}, 'http://127.0.0.1/')

    def test_compaction_keeps_original_if_empty_and_does_not_execute_tools(self):
        runner = self.runner()
        runner.append_user('Important requirements')
        original = json.loads(json.dumps(runner.state.messages))
        with patch.object(runner.client, 'call', return_value=reply('')) as client:
            with self.assertRaisesRegex(RuntimeError, 'no summary'):
                runner.compact()
            self.assertEqual(runner.state.messages, original)
            self.assertEqual(client.call_args.args[2], [])
        with patch.object(runner.client, 'call', return_value=reply('Preserved requirements')):
            runner.compact()
        self.assertIn('Preserved requirements', runner.state.messages[0]['content'][0]['text'])

    def test_api_request_uses_configured_model_tools_and_cache_usage(self):
        runner = self.runner()
        runner.client.model = 'custom/model'
        events = [
            {'type': 'message_start', 'message': {'model': 'custom/model', 'usage': {'input_tokens': 10, 'cache_read_input_tokens': 25}}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'OK'}},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
            {'type': 'message_stop'},
        ]
        data = io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))
        with patch('urllib.request.urlopen', return_value=data) as transport:
            result = runner.client.call([{'role': 'user', 'content': 'test'}], 'system', runner.schemas())
        request = transport.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, agent.API_URL)
        self.assertEqual(body['model'], 'custom/model')
        self.assertTrue(body['stream'])
        self.assertIn('write', {t['name'] for t in body['tools']})
        self.assertEqual(result['cached'], 25)
        self.assertEqual(result['context'], 36)


    def test_mid_task_compaction_restates_the_request_and_continues(self):
        self.config.max_context = 2000
        runner = self.runner()
        sent = []
        replies = [reply(calls=[tool("bash", {"cmd": "echo step1"})]), reply("Summary: step 1 done"), reply("finished")]
        def call(messages, system, tools, quiet=False):
            sent.append(json.loads(json.dumps(messages)))
            result = replies[len(sent) - 1]
            return {**result, "context": 1500}
        with patch.object(runner.client, "call", side_effect=call):
            self.assertEqual(runner.run("Do step 1, then step 2."), "finished")
        self.assertEqual(len(sent), 3)  # task, summary request, continuation
        after = sent[2]
        self.assertEqual(len(after), 1)
        text = after[0]["content"][0]["text"]
        self.assertIn("Summary: step 1 done", text)
        self.assertIn("Do step 1, then step 2.", text)  # the request survives verbatim
        self.assertIn("Continue it", text)
        self.assertNotIn("not a new task", text)
        # A manual compaction between tasks must not tell the model to keep working.
        with patch.object(runner.client, "call", return_value=reply("Idle summary")):
            runner.compact()
        idle = runner.state.messages[0]["content"][0]["text"]
        self.assertIn("not a new task", idle)
        self.assertNotIn("Continue it", idle)

    def test_estimate_counts_text_added_since_the_last_measurement(self):
        runner = self.runner()
        runner.append_user("x" * 100)
        runner.record_usage(reply("ok"))
        self.assertEqual(runner.estimate(), 110)
        runner.append_user("9" * 4000)
        self.assertGreater(runner.estimate(), 110 + 1900)  # 2 characters per token, not 3

    def test_shell_timeout_is_a_result_and_the_agent_keeps_working(self):
        for role in ("coordinator", "implementer"):
            runner = self.runner(role, owned_paths=["."])
            started = time.monotonic()
            result = runner.tool(tool("bash", {"cmd": "echo STARTED; sleep 30; echo ENDED", "timeout": 1}), set())
            self.assertLess(time.monotonic() - started, 5)
            self.assertTrue(result.startswith("error:"))
            self.assertIn("STARTED", result)
            self.assertNotIn("ENDED", result)
            self.assertIn("timed out after 1s", result)
            self.assertIn("next", runner.bash("echo next"))

    def test_shell_keeps_output_when_the_directory_is_not_allowed(self):
        (self.root / "node_modules").mkdir()
        (self.root / "sub").mkdir()
        runner = self.runner()
        for command in ("cd /tmp && echo KEPT", "cd node_modules && echo KEPT"):
            result = runner.bash(command)
            self.assertTrue(result.startswith("exit code 0"))
            self.assertIn("KEPT", result)
            self.assertIn("working directory stays", result)
            self.assertEqual(runner.state.cwd, str(self.root.resolve()))
        runner.bash("cd sub")
        self.assertEqual(runner.state.cwd, str(self.root.resolve() / "sub"))

    def test_long_shell_output_keeps_start_and_end_within_the_budget(self):
        runner = self.runner()
        result = runner.tool(tool("bash", {"cmd": "seq 1 20000; echo LAST-LINE"}), set())
        self.assertLessEqual(len(result), runner.output_limit)
        self.assertIn("\n1\n2\n", result)
        self.assertIn("LAST-LINE", result)
        self.assertIn("bytes cut", result)
        self.config.max_context = 4000
        self.assertEqual(self.runner().output_limit, 2000)

    def test_read_reports_empty_files_and_offsets_past_the_end(self):
        (self.root / "empty.txt").write_text("")
        (self.root / "one.txt").write_text("only\n")
        runner = self.runner()
        self.assertEqual(runner.read("empty.txt"), "(empty file)")
        self.assertEqual(runner.read("one.txt", offset=5), "(no lines at or after line 5)")
        self.assertIn("only", runner.read("one.txt"))



if __name__ == "__main__":
    unittest.main()
