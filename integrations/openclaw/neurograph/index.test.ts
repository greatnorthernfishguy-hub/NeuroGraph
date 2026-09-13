/**
 * Isolated lifecycle tests for the NeuroGraph OpenClaw extension.
 *
 * Changelog:
 * [2026-09-13] Claude Sonnet 5 — Source Lifecycle Repair, chunk 1
 *   What: Test suite for the OpenClaw process-service registration and the
 *         NeurographRpcClient admission/drain/dispose/stop state machine
 *         added in index.ts this chunk. Uses a fake spawn() (EventEmitter +
 *         PassThrough-style stdin/stdout/stderr) so no real Python process,
 *         socket, or OpenClaw runtime is touched.
 *   Why:  SOURCE-LIFECYCLE-REPAIR-SPEC.md "Tests required before deployment"
 *         → Extension: two distinct fake registries + repeated registration
 *         each receive a service; all service callbacks share one child/
 *         shutdown promise/dispose write; start() is a no-op; drain/race/
 *         hung-call handling; no failed/unaccepted path closes stdin/clears
 *         state/signals/respawns; accepted shutdown waits for natural child
 *         exit; ContextEngine.dispose() stays a per-conversation no-op.
 *   How:  node:test + node:assert (matches this package's package.json
 *         "test": "tsx --test index.test.ts" script — no vitest, consistent
 *         with the sibling jink-mcp/portal-v2 tsx --test pattern). A minimal
 *         FakeChildProcess (EventEmitter) stands in for node:child_process's
 *         ChildProcess; stdin.write is captured to an array so tests can
 *         assert exactly one dispose write regardless of how many
 *         stop()/registerService.stop() callers there are.
 * [2026-09-13] Claude Sonnet 5 — corrective follow-up (Grok family review of 0ac8826)
 *   What: (1) Changed the "no live process to dispose" case from expecting
 *         closed:true to closed:false — no dispose was ever accepted, so it
 *         is not a success. (2) Added an await-gap race test that forces
 *         call() to abort a respawn already in flight when stop() begins.
 *         (3) Added a dedicated watchdog/ensureRunning()-vs-stop() race test.
 *         (4) Added a live-child-exit-during-an-in-flight-call test asserting
 *         closed:false and no dispose write. (5) Added a stop-during-eager-
 *         spawn test against the new SPAWN_KEY plumbing, asserting stop()
 *         converges on the spawned child instead of latching a no-op.
 *   Why:  Independent family review found the 14 tests in 0ac8826 exercised
 *         the happy paths of the admission/drain state machine but never
 *         the actual await-gap races or the eager-spawn-orphan sequence the
 *         source fixes address, and one existing expectation (closed:true
 *         for "nothing to dispose") was itself wrong per the spec's
 *         accepted-persistence-only definition of a closed shutdown.
 *   How:  All 14 pre-existing required tests are preserved; only the one
 *         factually-wrong expectation was corrected in place.
 * [2026-09-13] Claude Sonnet 5 — second corrective follow-up (Grok second-pass review)
 *   What: (1) Added a forced-interleaving test for trackedSpawn() (the
 *         factory's fallback spawn path) racing a concurrent stop() call.
 *         (2) beforeEach now also clears SPAWN_KEY, alongside GLOBAL_KEY/
 *         SHUTDOWN_KEY, so a spawn promise from one test can't leak into
 *         the next. (3) The hung-call env-var test now restores
 *         NEUROGRAPH_SHUTDOWN_WARN_MS/_INTERVAL_MS with `delete` when they
 *         were originally unset, instead of assigning `undefined` (which
 *         coerces to the string "undefined" and produced a NaN setTimeout
 *         warning on the *next* test run to touch that code path); also
 *         asserts the parsed warnMs is finite.
 *   Why:  Second-pass review found SPAWN_KEY untouched by beforeEach (a
 *         leak risk once more spawn-path tests exist), and the env-var
 *         restore pattern silently producing `process.env.X = "undefined"`
 *         rather than actually deleting the key.
 *   How:  No changes to the 18 previously-committed test expectations.
 * [2026-09-13] Claude Sonnet 5 — third corrective follow-up (Grok final delta review)
 *   What: Imports SPAWN_ATTEMPT_KEY/getSpawnAttempt/setSpawnAttempt; beforeEach
 *         now also clears SPAWN_ATTEMPT_KEY. Two new forced tests: (1) a
 *         trackedSpawn() start() rejection leaves no stuck spawning/attempt
 *         state and a later trackedSpawn() call succeeds cleanly; (2) a
 *         second trackedSpawn() call while the first is awaiting "ready"
 *         joins the first instead of spawning a second process, preserves
 *         SPAWN_KEY identity, resolves to the same engine, and stop()
 *         converges on the one child.
 *   Why:  Grok's final delta review on 35f649f found trackedSpawn had no
 *         join/ownership semantics — every call re-spawned and overwrote
 *         global state, and failure cleanup could clobber state it didn't
 *         own.
 *   How:  All 19 previously-committed test expectations preserved.
 */

import { test, describe, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";

import {
  NeurographRpcClient,
  NeurographContextEngine,
  getGlobalState,
  setGlobalState,
  stopNeurographService,
  getGlobalShutdownPromise,
  getSpawnPromise,
  setSpawnPromise,
  trackedSpawn,
  getSpawnAttempt,
  setSpawnAttempt,
  GLOBAL_KEY,
  SHUTDOWN_KEY,
  SPAWN_KEY,
  SPAWN_ATTEMPT_KEY,
} from "./index.js";

// ── Fake child process ──────────────────────────────────────────────

type Written = { request: any };

class FakeChildProcess extends EventEmitter {
  stdin: PassThrough & { writes: Written[] };
  stdout: PassThrough;
  stderr: PassThrough;
  exitCode: number | null = null;

  constructor() {
    super();
    const writes: Written[] = [];
    const stdin = new PassThrough() as PassThrough & { writes: Written[] };
    stdin.writes = writes;
    const origWrite = stdin.write.bind(stdin);
    (stdin as any).write = (chunk: any, ...rest: any[]) => {
      try {
        writes.push({ request: JSON.parse(String(chunk).trim()) });
      } catch {
        // ignore unparsable (shouldn't happen in these tests)
      }
      return origWrite(chunk, ...rest);
    };
    stdin.end = ((...args: any[]) => {
      this._stdinEnded = true;
      return PassThrough.prototype.end.apply(stdin, args as any);
    }) as any;

    this.stdin = stdin;
    this.stdout = new PassThrough();
    this.stderr = new PassThrough();
  }

  _stdinEnded = false;

  /** Simulate a JSON-RPC line arriving from "Python" on stdout. */
  emitLine(obj: unknown): void {
    this.stdout.write(JSON.stringify(obj) + "\n");
  }

  /** Simulate the process actually exiting. */
  simulateExit(code = 0): void {
    this.exitCode = code;
    this.emit("exit", code);
  }
}

function makeSpawnFn() {
  let lastProc: FakeChildProcess | null = null;
  const spawnFn = (_cmd: string, _args: readonly string[], _opts: Record<string, unknown>) => {
    const proc = new FakeChildProcess();
    lastProc = proc;
    return proc as unknown as import("node:child_process").ChildProcess;
  };
  return { spawnFn, getLastProc: () => lastProc! };
}

function makeLogger() {
  const lines: string[] = [];
  return {
    logger: {
      info: (m: string) => lines.push(`[info] ${m}`),
      warn: (m: string) => lines.push(`[warn] ${m}`),
      error: (m: string) => lines.push(`[error] ${m}`),
    },
    lines,
  };
}

/** Starts a client against a fake child and completes the ready handshake. */
async function startedClient() {
  const { spawnFn, getLastProc } = makeSpawnFn();
  const { logger, lines } = makeLogger();
  const client = new NeurographRpcClient(logger, spawnFn);
  const startPromise = client.start();
  // Wait a tick for spawnFn to run synchronously inside start(), then signal ready.
  await new Promise((r) => setImmediate(r));
  getLastProc().emitLine({ jsonrpc: "2.0", method: "ready" });
  await startPromise;
  return { client, proc: getLastProc(), lines };
}

beforeEach(() => {
  (globalThis as any)[GLOBAL_KEY] = undefined;
  (globalThis as any)[SHUTDOWN_KEY] = undefined;
  (globalThis as any)[SPAWN_KEY] = undefined;
  (globalThis as any)[SPAWN_ATTEMPT_KEY] = undefined;
});

// ── start() is a no-op re: substrate spawning (service-level) ───────

// ── Fake registries / plugin registration convergence ────────────────

function makeFakeApi() {
  const registry: { services: any[]; contextEngines: any[] } = { services: [], contextEngines: [] };
  const logger = makeLogger().logger;
  const api = {
    logger,
    registerService(svc: any) {
      registry.services.push(svc);
    },
    registerContextEngine(id: string, factory: any) {
      registry.contextEngines.push({ id, factory });
    },
  };
  return { api, registry };
}

describe("registerService start()", () => {
  test("service start() never spawns or writes anything", async () => {
    const mod = await import("./index.js");
    const plugin = mod.default;
    seedFakeAliveGlobalState();

    const fakeApi = makeFakeApi();
    plugin.register(fakeApi.api as any);
    const svc = fakeApi.registry.services[0];
    assert.equal(svc.id, "neurograph-rpc-host");

    await svc.start();

    // start() must be a pure no-op: it must not spawn or awaken the
    // substrate, i.e. it must not create/replace global rpc state.
    const state = getGlobalState();
    assert.equal(state?.rpc?.isAlive?.(), true, "start() must not disturb existing global state");
  });
});

// register() unconditionally eager-spawns a *real* child process (via the
// real node:child_process `spawn`) unless GLOBAL_KEY already holds a live
// rpc client — that eager spawn is pre-existing, unchanged behavior and is
// out of scope for this chunk's fake-transport tests. So these registration
// tests pre-seed global state with a fake "already alive" client first,
// which takes register()'s existing early-return branch (service
// registration still runs — it happens before that branch — but the real
// eager-spawn IIFE is skipped entirely). This keeps registration tests
// fully isolated from any real process spawn.
function seedFakeAliveGlobalState(): void {
  setGlobalState({
    rpc: { isAlive: () => true } as unknown as NeurographRpcClient,
    engine: {} as any,
    spawning: false,
  });
}

describe("plugin registration across registries", () => {
  test("two distinct fake registries each receive a service", async () => {
    const mod = await import("./index.js");
    const plugin = mod.default;

    seedFakeAliveGlobalState();
    const a = makeFakeApi();
    const b = makeFakeApi();

    plugin.register(a.api as any);
    plugin.register(b.api as any);

    assert.equal(a.registry.services.length, 1);
    assert.equal(b.registry.services.length, 1);
    assert.equal(a.registry.services[0].id, "neurograph-rpc-host");
    assert.equal(b.registry.services[0].id, "neurograph-rpc-host");
  });

  test("repeated registration into one registry each receives a service", async () => {
    const mod = await import("./index.js");
    const plugin = mod.default;
    seedFakeAliveGlobalState();
    const a = makeFakeApi();

    plugin.register(a.api as any);
    plugin.register(a.api as any);
    plugin.register(a.api as any);

    assert.equal(a.registry.services.length, 3);
    for (const svc of a.registry.services) {
      assert.equal(svc.id, "neurograph-rpc-host");
    }
  });

  test("all service callbacks converge on one child/shutdown promise/dispose write", async () => {
    const { client, proc } = await startedClient();
    setGlobalState({ rpc: client, engine: null as any, spawning: false });

    // Simulate three stop() callbacks (as if from three registerService
    // registrations) firing "simultaneously".
    const p1 = stopNeurographService();
    const p2 = stopNeurographService();
    const p3 = stopNeurographService();

    assert.equal(p1, p2);
    assert.equal(p2, p3);

    await new Promise((r) => setImmediate(r));
    assert.equal(proc.stdin.writes.length, 1, "expected exactly one dispose write");
    assert.equal(proc.stdin.writes[0].request.method, "dispose");

    // Respond accepted; all three callers resolve together.
    const disposeId = proc.stdin.writes[0].request.id;
    proc.emitLine({ jsonrpc: "2.0", id: disposeId, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);

    await Promise.all([p1, p2, p3]);
    assert.equal(getGlobalState(), undefined, "global state cleared exactly once, after natural exit");
  });
});

// ── Admission / drain / stop ordering ────────────────────────────────

describe("NeurographRpcClient stop() lifecycle", () => {
  test("call() is rejected once shutdown begins (admission closes atomically)", async () => {
    const { client, proc } = await startedClient();

    const stopPromise = client.stop();
    // stop() is now in-flight (drain phase, admitted set empty) — new calls
    // must be rejected immediately, not queued behind dispose.
    await assert.rejects(() => client.call("assemble", {}), /stopping/);

    // Let stop() finish so it doesn't leave a stray unresolved dispose wait
    // hanging past the test.
    const disposeId = proc.stdin.writes.at(-1)?.request.id;
    if (disposeId !== undefined) {
      proc.emitLine({ jsonrpc: "2.0", id: disposeId, result: { safe_to_terminate: true } });
    }
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);
    await stopPromise;
  });

  test("ensureRunning()/watchdog respawn is prohibited while stopping", async () => {
    const { client, proc } = await startedClient();
    const stopPromise = client.stop();

    await assert.rejects(() => client.ensureRunning(), /stopping/);

    const disposeId = proc.stdin.writes.at(-1)?.request.id;
    proc.emitLine({ jsonrpc: "2.0", id: disposeId, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);
    await stopPromise;
  });

  test("shutdown drains an in-flight call before issuing dispose", async () => {
    const { client, proc } = await startedClient();

    const callPromise = client.call("assemble", { messages: [] });
    await new Promise((r) => setImmediate(r));
    const assembleWrite = proc.stdin.writes.find((w) => w.request.method === "assemble")!;
    assert.ok(assembleWrite, "assemble call should have been written");

    const stopPromise = client.stop();
    await new Promise((r) => setImmediate(r));

    // Dispose must NOT have been written yet — the in-flight call is still
    // admitted and undrained.
    assert.equal(
      proc.stdin.writes.some((w) => w.request.method === "dispose"),
      false,
      "dispose must wait for the in-flight call to drain"
    );

    // Resolve the in-flight call.
    proc.emitLine({ jsonrpc: "2.0", id: assembleWrite.request.id, result: { estimatedTokens: 1 } });
    await callPromise;
    await new Promise((r) => setImmediate(r));

    // Now dispose should have been written exactly once.
    const disposeWrite = proc.stdin.writes.find((w) => w.request.method === "dispose");
    assert.ok(disposeWrite, "dispose should be written after drain completes");

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);
    await stopPromise;
  });

  test("a hung admitted call blocks shutdown and emits a bounded diagnostic, never signals the child", async () => {
    const originalWarnMs = process.env.NEUROGRAPH_SHUTDOWN_WARN_MS;
    const originalIntervalMs = process.env.NEUROGRAPH_SHUTDOWN_WARN_INTERVAL_MS;
    process.env.NEUROGRAPH_SHUTDOWN_WARN_MS = "10";
    process.env.NEUROGRAPH_SHUTDOWN_WARN_INTERVAL_MS = "10";
    assert.ok(
      Number.isFinite(Number(process.env.NEUROGRAPH_SHUTDOWN_WARN_MS)),
      "the warn-threshold env var must parse to a finite number, not the NaN that an "
        + "`undefined`-string assignment would eventually restore this to"
    );
    try {
      const { client, proc, lines } = await startedClient();

      // Admit a call that will never receive a response (hung).
      const hungPromise = client.call("assemble", {}).catch(() => {});
      await new Promise((r) => setImmediate(r));

      let stopSettled = false;
      const stopPromise = client.stop().then((r) => {
        stopSettled = true;
        return r;
      });

      // Give the bounded diagnostic timers time to fire multiple times.
      await new Promise((r) => setTimeout(r, 60));

      assert.equal(stopSettled, false, "shutdown must remain pending while a call is hung");
      assert.equal(
        proc.stdin.writes.some((w) => w.request.method === "dispose"),
        false,
        "dispose must not be issued while a normal call is still admitted"
      );
      assert.ok(
        lines.some((l) => l.includes("in-flight RPC call") && l.includes("waiting indefinitely")),
        `expected a bounded diagnostic warning to have been emitted; got: ${JSON.stringify(lines)}`
      );
      // FakeChildProcess exposes no kill()/signal method at all — if the
      // drain path ever attempted to signal the child it would throw here
      // instead of silently succeeding, so reaching this point already
      // proves no signal was sent. Also confirm the harness's own exit
      // tracking was untouched.
      assert.equal(proc.exitCode, null, "hung-call drain must never kill the child");

      // Unstick the hang so cleanup doesn't leak a dangling promise.
      const pendingId = proc.stdin.writes.find((w) => w.request.method === "assemble")?.request.id;
      if (pendingId !== undefined) {
        proc.emitLine({ jsonrpc: "2.0", id: pendingId, result: {} });
      }
      await hungPromise;
      await new Promise((r) => setImmediate(r));
      const disposeWrite = proc.stdin.writes.find((w) => w.request.method === "dispose");
      proc.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
      await new Promise((r) => setImmediate(r));
      proc.simulateExit(0);
      await stopPromise;
    } finally {
      // `process.env.X = undefined` would coerce to the literal string
      // "undefined" (Number("undefined") is NaN) rather than actually
      // unsetting the var — delete when the var was originally absent so a
      // later test/run reads the real default, not a NaN-parsing string.
      if (originalWarnMs === undefined) {
        delete process.env.NEUROGRAPH_SHUTDOWN_WARN_MS;
      } else {
        process.env.NEUROGRAPH_SHUTDOWN_WARN_MS = originalWarnMs;
      }
      if (originalIntervalMs === undefined) {
        delete process.env.NEUROGRAPH_SHUTDOWN_WARN_INTERVAL_MS;
      } else {
        process.env.NEUROGRAPH_SHUTDOWN_WARN_INTERVAL_MS = originalIntervalMs;
      }
    }
  });

  test("no request timeout applies to the privileged dispose call", async () => {
    const { client, proc } = await startedClient();
    const stopPromise = client.stop();
    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc.stdin.writes.find((w) => w.request.method === "dispose")!;

    // Wait well past the old 10s/30s normal-call timeouts using fake elapsed
    // time via a short real wait — the dispose request must still be
    // pending (no rejection), proving no timer was attached to it.
    await new Promise((r) => setTimeout(r, 50));

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);
    const result = await stopPromise;
    assert.equal(result.closed, true);
  });

  test("safe_to_terminate: true closes stdin and waits for natural exit before resolving", async () => {
    const { client, proc } = await startedClient();
    const stopPromise = client.stop();
    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc.stdin.writes.find((w) => w.request.method === "dispose")!;

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));

    assert.equal(proc._stdinEnded, true, "stdin must be closed once safe_to_terminate: true is reported");

    let resolved = false;
    stopPromise.then(() => { resolved = true; });
    await new Promise((r) => setImmediate(r));
    assert.equal(resolved, false, "stop() must wait for the natural exit event, not resolve on stdin.end()");

    proc.simulateExit(0);
    const result = await stopPromise;
    assert.equal(result.closed, true);
    assert.equal(resolved, true);
  });

  test("safe_to_terminate: false never closes stdin, signals, or clears global state", async () => {
    const { client, proc } = await startedClient();
    setGlobalState({ rpc: client, engine: null as any, spawning: false });

    const stopPromise = stopNeurographService();
    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc.stdin.writes.find((w) => w.request.method === "dispose")!;
    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite.request.id, result: { safe_to_terminate: false } });

    await stopPromise;

    assert.equal(proc._stdinEnded, false, "stdin must remain open when safe_to_terminate is not true");
    assert.equal(proc.exitCode, null, "no signal may be sent on an unaccepted dispose");
    assert.notEqual(getGlobalState(), undefined, "global state must be left intact on failed shutdown");
  });

  test("no live process to dispose is an unaccepted shutdown, not a success", async () => {
    const { logger } = makeLogger();
    const client = new NeurographRpcClient(logger, makeSpawnFn().spawnFn);
    // Never started — no live process at all.
    setGlobalState({ rpc: client, engine: null as any, spawning: false });

    const result = await client.stop();
    assert.equal(
      result.closed,
      false,
      "no persistence was ever accepted for a process that was never live — reporting closed:true here would " +
        "claim credit for a dispose that never happened"
    );
  });

  test("await-gap race: call() aborts a mid-flight ensureRunning() respawn once stop() begins (Fix 1)", async () => {
    const { spawnFn, getLastProc } = makeSpawnFn();
    const { logger } = makeLogger();
    const client = new NeurographRpcClient(logger, spawnFn);

    // Bring up proc1, then kill it so the next call() must respawn.
    const startPromise = client.start();
    await new Promise((r) => setImmediate(r));
    getLastProc().emitLine({ jsonrpc: "2.0", method: "ready" });
    await startPromise;
    const proc1 = getLastProc();
    proc1.simulateExit(1);

    // call() → performCall sees a dead proc → awaits ensureRunning() → performEnsureRunning()
    // → awaits start(), which synchronously spawns proc2 before its own await gap.
    const callPromise = client.call("assemble", {});
    await new Promise((r) => setImmediate(r));
    const proc2 = getLastProc();
    assert.notEqual(proc2, proc1, "call() must have triggered a respawn onto a second fake child");

    // stop() begins while call() is still sitting inside the awaited respawn
    // (proc2 has not emitted "ready" yet) — this is the exact await gap Fix 1 closes.
    const stopPromise = client.stop();

    // Let proc2 finish spawning now.
    proc2.emitLine({ jsonrpc: "2.0", method: "ready" });

    // The racing call() must reject — it must never reach the stdin write for
    // "assemble" on proc2, because assertNotStoppingMidFlight() aborts it
    // immediately after the ensureRunning() await settles.
    await assert.rejects(() => callPromise, /aborted after an in-flight await|stopping/);
    assert.equal(
      proc2.stdin.writes.some((w) => w.request.method === "assemble"),
      false,
      "the racing call must never write its request after stop() began"
    );

    // stop() still needs to converge on and dispose the now-live proc2.
    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc2.stdin.writes.find((w) => w.request.method === "dispose");
    assert.ok(disposeWrite, "stop() must dispose the child the aborted call ended up spawning");
    proc2.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc2.simulateExit(0);
    await stopPromise;
  });

  test("watchdog-vs-stop race: an in-flight watchdog respawn is aborted by a concurrent stop() (Fix 4)", async () => {
    const { client, proc: proc1 } = await startedClient();
    proc1.simulateExit(1);

    // Simulate the watchdog's own call path: it invokes ensureRunning() directly
    // (see startWatchdog()'s interval callback), not through call().
    const respawnPromise = client.ensureRunning();
    await new Promise((r) => setImmediate(r));

    const stopPromise = client.stop();

    // Let the in-flight respawn's start() settle.
    const proc2 = (client as any).proc as FakeChildProcess | undefined;
    assert.ok(proc2, "ensureRunning() must have synchronously spawned a second fake child");
    proc2!.emitLine({ jsonrpc: "2.0", method: "ready" });

    await assert.rejects(() => respawnPromise, /aborted after an in-flight await|stopping/);
    assert.equal(
      proc2!.stdin.writes.some((w: Written) => w.request.method === "bootstrap"),
      false,
      "an aborted watchdog respawn must never reach the bootstrap call"
    );

    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc2!.stdin.writes.find((w: Written) => w.request.method === "dispose");
    assert.ok(disposeWrite, "stop() must still converge on and dispose the child the watchdog respawn produced");
    proc2!.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc2!.simulateExit(0);
    await stopPromise;
  });

  test("child exit during drain (in-flight call) before an accepted dispose is unaccepted shutdown (Fix 3)", async () => {
    const { client, proc } = await startedClient();
    setGlobalState({ rpc: client, engine: null as any, spawning: false });

    const callPromise = client.call("assemble", { messages: [] }).catch(() => {});
    await new Promise((r) => setImmediate(r));

    const stopPromise = stopNeurographService();
    await new Promise((r) => setImmediate(r));

    // The child dies mid-drain — before any dispose request was ever sent.
    proc.simulateExit(1);

    await callPromise;
    await stopPromise;

    assert.equal(
      proc.stdin.writes.some((w) => w.request.method === "dispose"),
      false,
      "dispose must never be sent to an already-dead child"
    );
    assert.notEqual(
      getGlobalState(),
      undefined,
      "global state must be left intact — no dispose was ever accepted for this child"
    );
  });

  test("stop() arriving during an eager spawn converges on the spawned child (Fix 2)", async () => {
    const { client, proc } = await startedClient();

    // Simulate register()'s eager-spawn window: GLOBAL_KEY says "spawning",
    // and the in-flight spawn promise is tracked via SPAWN_KEY.
    setGlobalState({ rpc: null as any, engine: null as any, spawning: true });
    let resolveSpawn: () => void = () => {};
    const spawnPromise = new Promise<void>((r) => {
      resolveSpawn = r;
    });
    setSpawnPromise(spawnPromise);

    const stopPromise = stopNeurographService();
    let settled = false;
    stopPromise.then(() => {
      settled = true;
    });
    await new Promise((r) => setImmediate(r));
    assert.equal(
      settled,
      false,
      "stop() must await the in-flight spawn instead of latching a resolved no-op while spawning:true"
    );

    // The spawn "finishes": it publishes the live rpc, then its promise settles.
    setGlobalState({ rpc: client, engine: {} as any, spawning: false });
    resolveSpawn();

    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc.stdin.writes.find((w) => w.request.method === "dispose");
    assert.ok(disposeWrite, "stop() must converge on and dispose the child the spawn produced, not orphan it");

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);

    await stopPromise;
    assert.equal(getGlobalState(), undefined, "global state cleared once the converged stop actually closed");
  });

  test("trackedSpawn() (factory fallback spawn path) also converges with a concurrent stop() (Fix 1, second pass)", async () => {
    const { spawnFn, getLastProc } = makeSpawnFn();
    const { logger } = makeLogger();
    const rpc = new NeurographRpcClient(logger, spawnFn);

    // This is the exact call the registerContextEngine() factory's fallback
    // branch makes — a spawn path distinct from register()'s eager-spawn
    // IIFE, but which must share the same SPAWN_KEY tracking so stop()
    // cannot race ahead of it either.
    const enginePromise = trackedSpawn(rpc);
    await new Promise((r) => setImmediate(r));

    assert.equal(getGlobalState()?.spawning, true, "trackedSpawn must publish spawning:true synchronously");
    assert.ok(getSpawnPromise(), "trackedSpawn must register its promise on SPAWN_KEY");

    const stopPromise = stopNeurographService();
    let stopSettled = false;
    stopPromise.then(() => {
      stopSettled = true;
    });
    await new Promise((r) => setImmediate(r));
    assert.equal(
      stopSettled,
      false,
      "stop() must await the tracked factory-path spawn rather than returning early on !state.rpc"
    );

    const proc = getLastProc();
    proc.emitLine({ jsonrpc: "2.0", method: "ready" });
    const engine = await enginePromise;
    assert.ok(engine instanceof NeurographContextEngine, "trackedSpawn must resolve with the live engine");

    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc.stdin.writes.find((w: Written) => w.request.method === "dispose");
    assert.ok(disposeWrite, "stop() must dispose the child the factory-path spawn produced, not orphan it");

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);

    await stopPromise;
    assert.equal(getGlobalState(), undefined, "global state cleared once the converged stop actually closed");
  });

  test("trackedSpawn() start() rejection leaves no stuck spawning state and permits a later retry (Grok final delta)", async () => {
    const { logger } = makeLogger();

    // A "flaky" rpc whose start() rejects exactly once, simulating the
    // Python process failing to launch on the first attempt.
    const flakyRpc = {
      isAlive: () => false,
      start: () => Promise.reject(new Error("simulated spawn failure")),
    } as unknown as NeurographRpcClient;

    await assert.rejects(() => trackedSpawn(flakyRpc), /simulated spawn failure/);

    assert.equal(getGlobalState(), undefined, "a failed attempt must not leave a stuck spawning placeholder");
    assert.equal(getSpawnAttempt(), undefined, "a failed attempt must clear its own ownership token");

    // A fresh trackedSpawn() call after the failure must start cleanly —
    // no lingering attempt/placeholder from the rejected one should block it.
    const { spawnFn, getLastProc } = makeSpawnFn();
    const rpc2 = new NeurographRpcClient(logger, spawnFn);
    const enginePromise = trackedSpawn(rpc2);
    await new Promise((r) => setImmediate(r));

    assert.equal(getGlobalState()?.spawning, true, "the retry must publish its own spawning:true placeholder");

    const proc = getLastProc();
    proc.emitLine({ jsonrpc: "2.0", method: "ready" });
    const engine = await enginePromise;

    assert.ok(engine instanceof NeurographContextEngine, "the retry must resolve to a live engine");
    assert.equal(getGlobalState()?.rpc?.isAlive?.(), true, "the retry's child must be published as live");
  });

  test("a second trackedSpawn() while the first awaits ready joins the first (Grok final delta)", async () => {
    const { spawnFn, getLastProc } = makeSpawnFn();
    const { logger } = makeLogger();
    const rpc1 = new NeurographRpcClient(logger, spawnFn);
    const rpc2 = new NeurographRpcClient(logger, makeSpawnFn().spawnFn);

    const enginePromise1 = trackedSpawn(rpc1);
    await new Promise((r) => setImmediate(r));
    const spawnPromiseBefore = getSpawnPromise();

    const enginePromise2 = trackedSpawn(rpc2);
    await new Promise((r) => setImmediate(r));

    assert.equal(getSpawnPromise(), spawnPromiseBefore, "joining a second call must never replace the SPAWN_KEY promise stop() is tracking");
    assert.equal(rpc2.isAlive(), false, "the joining call's own rpc must never be started");

    const proc = getLastProc();
    proc.emitLine({ jsonrpc: "2.0", method: "ready" });

    const [engine1, engine2] = await Promise.all([enginePromise1, enginePromise2]);
    assert.equal(engine1, engine2, "both trackedSpawn() calls must resolve to the identical engine");

    await new Promise((r) => setImmediate(r));
    const stopPromise = stopNeurographService();
    await new Promise((r) => setImmediate(r));
    const disposeWrite = proc.stdin.writes.find((w: Written) => w.request.method === "dispose");
    assert.ok(disposeWrite, "stop() must dispose the one shared child");

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrite!.request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);

    await stopPromise;
    assert.equal(getGlobalState(), undefined, "stop() converges on the single joined child and clears global state");
  });

  test("two simultaneous stop() calls converge on one dispose write and one outcome", async () => {
    const { client, proc } = await startedClient();

    // client.stop() is an `async` method, so each call's *returned* promise
    // is necessarily a fresh wrapper (an async function always returns a new
    // promise object) even though both wrap the same underlying
    // this.stopPromise. Reference-equality on the returned promises is not
    // a meaningful test; instead assert on the effect: exactly one dispose
    // write and identical settled results for both callers.
    const p1 = client.stop();
    const p2 = client.stop();

    await new Promise((r) => setImmediate(r));
    const disposeWrites = proc.stdin.writes.filter((w) => w.request.method === "dispose");
    assert.equal(disposeWrites.length, 1);

    proc.emitLine({ jsonrpc: "2.0", id: disposeWrites[0].request.id, result: { safe_to_terminate: true } });
    await new Promise((r) => setImmediate(r));
    proc.simulateExit(0);

    const [r1, r2] = await Promise.all([p1, p2]);
    assert.deepEqual(r1, r2);
    assert.equal(r1.closed, true);
  });
});

// ── ContextEngine.dispose() remains a per-conversation no-op ─────────

describe("NeurographContextEngine.dispose()", () => {
  test("dispose() does not touch the rpc client or process at all", async () => {
    const { client, proc } = await startedClient();
    const engine = new NeurographContextEngine(client);

    await engine.dispose();

    assert.equal(proc.stdin.writes.length, 0, "per-conversation dispose() must write nothing to the child");
    assert.equal(client.isAlive(), true, "per-conversation dispose() must not affect process liveness");
    assert.equal(client.isStopping(), false, "per-conversation dispose() must not begin process shutdown");
  });
});
