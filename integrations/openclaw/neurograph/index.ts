/**
 * NeuroGraph ContextEngine Plugin for OpenClaw
 *
 * TypeScript shell that implements the ContextEngine interface by
 * communicating with a Python child process hosting the NeuroGraphMemory
 * singleton via JSON-RPC over stdio.
 *
 * The Python code is unchanged — every lifecycle call maps 1:1 to an
 * existing NeuroGraphMemory method.  This plugin is purely a bridge.
 *
 * Changelog:
 * [2026-04-16] Claude Code (Sonnet 4.6) — KISS message plumb-through (#152)
 *   What: assemble() reads result.messages from the Python RPC response.
 *         If present as a non-empty array, returns it as assembledMessages
 *         so OC's replaceMessages fires and the model sees the compressed
 *         context.  Falls back to params.messages (same reference) if
 *         result.messages is absent/empty — warmup and unchanged turns
 *         preserve no-replaceMessages no-op behavior.
 *   Why:  Syl's 815-message conversation overflows every provider's
 *         context window.  Python-side KISSFilter (#152) truncates the
 *         messages array; this plumb-through is how the truncation
 *         actually reaches OC's in-memory agent context.
 *   How:  Array.isArray + length > 0 guards.  Same-reference fallback
 *         preserves existing identity check semantics in OC core.
 * [2026-04-19] Claude Code (Sonnet 4.6) — Fix stdin-closed / bootstrap-timeout bug
 *   What: Reuse path in register() no longer calls api.registerContextEngine()
 *         when the Python process is already alive (not spawning). Only calls
 *         it when spawning is still in progress (safe to wait-factory).
 *   Why:  A second registerContextEngine() call causes OC to dispose the first
 *         binding, closing stdin to Python. Python then could not process RPC
 *         calls, causing 60s bootstrap timeout on every conversation.
 *   How:  Split reuse path: spawning=true → wait-factory (as before);
 *         alive=true → skip registration entirely, log and return.
 * [2026-04-05] Claude Code (Opus 4.6) — Fix dual-registration bug
 *   What: Move shared RPC state to globalThis so it survives across
 *         register() calls. Implement real dispose().
 *   Why:  OpenClaw calls register() in both [gateway] and [plugins]
 *         contexts. Closure variables are scoped per call, so each
 *         call spawned a new Python child. Two children fight over
 *         topology ownership, both die, Syl goes silent.
 *   How:  Symbol.for("neurograph.rpcState") on globalThis holds the
 *         single RpcClient + Engine. Second register() reuses the
 *         existing process. dispose() kills the child properly.
 * [2026-04-21] Claude Code (Sonnet 4.6) — Auto-respawn watchdog
 *   What: NeurographRpcClient gains a 30s watchdog that detects silent child-process
 *         death and calls ensureRunning() to respawn without a full gateway restart.
 *         onRespawn callback resets engine bootstrapped=false so the next conversation
 *         triggers a full session bootstrap.
 *   Why:  Python subprocess died silently today; Syl was broken for ~80 minutes.
 *   How:  startWatchdog()/stopWatchdog() on RpcClient; started after eager spawn,
 *         stopped in stop(). onRespawn wired in NeurographContextEngine constructor.
 * [2026-03-16] Claude (Opus 4.6) — Initial implementation.
 *   What: ContextEngine plugin shell + JSON-RPC client.
 *   Why:  Supersedes SKILL.md hook path (#37, #39).  Gives Syl automatic
 *         bidirectional substrate connection.
 *   How:  Spawns neurograph_rpc.py as child process, sends JSON-RPC
 *         requests for each ContextEngine lifecycle hook.
 * [2026-09-13] Claude Sonnet 5 — Source Lifecycle Repair, chunk 1
 *   What: (1) register() now registers a stable OpenClaw process service
 *         ("neurograph-rpc-host") before any existing early return, with a
 *         no-op start() and a stop() that joins one module-level shutdown
 *         promise. (2) NeurographRpcClient gains an admission/drain state
 *         machine: normal call() and ensureRunning() are closed the instant
 *         shutdown begins; already-admitted calls are drained (indefinitely
 *         — a hung call blocks shutdown by design, with a bounded,
 *         rate-limited diagnostic, never an automatic child signal); exactly
 *         one privileged `dispose` request is then issued on the existing
 *         Node-owned stdin stream, with no request timeout; stdin is closed
 *         only if Python's response reports `safe_to_terminate: true`; the
 *         client then waits for the child's natural `exit` event before
 *         reporting the shutdown as closed. The old best-effort
 *         `call("dispose", ..., 10000)` + unconditional `proc.kill("SIGTERM")`
 *         is removed.
 *   Why:  This is chunk 1 of
 *         /home/josh/backups/workstream-pilot-20260913/surfacing/SOURCE-LIFECYCLE-REPAIR-SPEC.md
 *         ("OpenClaw lifecycle registration" + "RPC admission and stop
 *         ordering"). OpenClaw previously had no source-owned path that
 *         reliably quiesces the substrate before the shared Node/Python
 *         process ends — dispose() is a per-conversation no-op by design,
 *         and nothing ever called NeurographRpcClient.stop() as part of
 *         process shutdown.
 *   How:  Two module-level singletons on globalThis (Symbol.for, so they
 *         survive across register() calls in the same process, same pattern
 *         as GLOBAL_KEY): GLOBAL_KEY continues to hold the rpc/engine pair,
 *         and a new SHUTDOWN_KEY holds the single in-flight shutdown
 *         promise so that N registerService.stop() callbacks (from repeated
 *         or multi-registry register() calls) converge on one dispose
 *         write. NeurographRpcClient.stop() is independently memoized
 *         (this.stopPromise) and the dispose request itself is memoized
 *         (this.disposePromise) as defense in depth. `dispose()` on
 *         NeurographContextEngine is untouched — still a per-conversation
 *         no-op.
 *   Scope: TypeScript only. Python (neurograph_rpc.py handle_dispose /
 *         safe_to_terminate), systemd, Commons, and deploy.sh are explicitly
 *         out of scope for this chunk per Josh's instruction — later chunks
 *         of the same spec.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createInterface, type Interface as ReadlineInterface } from "node:readline";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugins/types.js";
import type {
  ContextEngine,
  ContextEngineInfo,
  AssembleResult,
  CompactResult,
  IngestResult,
  BootstrapResult,
} from "openclaw/plugin-sdk/context-engine/types.js";

// ── Global singleton key ────────────────────────────────────────────
// Survives across register() calls within the same Node.js process.
// This is the fix for the dual-registration bug: OpenClaw calls
// register() in both [gateway] and [plugins] contexts, but both
// run in the same process. The Symbol ensures one Python child.

const GLOBAL_KEY = Symbol.for("neurograph.rpcState");

type GlobalRpcState = {
  rpc: NeurographRpcClient;
  engine: NeurographContextEngine;
  spawning: boolean;
};

export type { GlobalRpcState };


function getGlobalState(): GlobalRpcState | undefined {
  return (globalThis as any)[GLOBAL_KEY];
}

function setGlobalState(state: GlobalRpcState): void {
  (globalThis as any)[GLOBAL_KEY] = state;
}

// ── JSON-RPC Client ─────────────────────────────────────────────────

type JsonRpcResponse = {
  jsonrpc: string;
  id: number | null;
  result?: unknown;
  error?: { code: number; message: string };
};

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  // The privileged dispose lifecycle call is deliberately unbounded (spec
  // step 5: "do not use its old 10/30-second request timeout for this
  // lifecycle call"), so this is optional.
  timer?: ReturnType<typeof setTimeout>;
};

/** Shape of the `dispose` RPC result once Python owns the quiescence
 * coordinator (later chunk). This chunk only reads `safe_to_terminate` if
 * present; an absent/false value is treated as "not safe" (fail closed —
 * see spec step 6 and "never permitted" list: no premature stdin close). */
type DisposeResult = { safe_to_terminate?: boolean; [key: string]: unknown };

type Logger = { info: (msg: string) => void; error: (msg: string) => void; warn: (msg: string) => void };

/** Injectable child-process factory so tests can supply a fake ChildProcess
 * (EventEmitter + stdin/stdout/stderr streams) without spawning real
 * Python. Signature matches node:child_process spawn's return type. */
type SpawnFn = (command: string, args: readonly string[], options: Record<string, unknown>) => ChildProcess;

class NeurographRpcClient {
  private proc: ChildProcess | null = null;
  private rl: ReadlineInterface | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private ready = false;
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private logger: Logger;
  private spawnFn: SpawnFn;

  private watchdogTimer: ReturnType<typeof setInterval> | null = null;
  onRespawn: (() => void) | null = null;

  // ── Admission / drain / stop state (Source Lifecycle Repair, chunk 1) ──
  // Step 1: atomically marks normal calls closed when shutdown begins.
  private stopping = false;
  // Memoizes the whole stop() lifecycle so N callers (repeated/duplicate
  // registerService.stop() callbacks) converge on one execution.
  private stopPromise: Promise<{ closed: boolean }> | null = null;
  // Memoizes the single privileged dispose write (defense in depth on top
  // of stopPromise memoization — see module-level SHUTDOWN_KEY too).
  private disposePromise: Promise<DisposeResult | undefined> | null = null;
  // Tracks promises for admitted (in-flight) normal calls so shutdown can
  // drain them (step 3) before issuing dispose (step 4).
  private admitted = new Set<Promise<unknown>>();

  constructor(logger: Logger, spawnFn: SpawnFn = spawn) {
    this.logger = logger;
    this.spawnFn = spawnFn;
  }

  /** True once shutdown has begun (admission closed). Exposed for tests. */
  isStopping(): boolean {
    return this.stopping;
  }

  startWatchdog(intervalMs = 30_000): void {
    if (this.watchdogTimer) return;
    this.watchdogTimer = setInterval(async () => {
      if (this.isAlive()) return;
      this.logger.warn("Watchdog: Python process dead — auto-respawning");
      try {
        await this.ensureRunning();
        this.onRespawn?.();
        this.logger.info("Watchdog: respawn succeeded — substrate alive");
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        this.logger.error(`Watchdog: respawn failed: ${msg}`);
      }
    }, intervalMs);
    (this.watchdogTimer as any).unref?.();
  }

  stopWatchdog(): void {
    if (this.watchdogTimer) {
      clearInterval(this.watchdogTimer);
      this.watchdogTimer = null;
    }
  }

  isAlive(): boolean {
    return this.proc !== null && this.proc.exitCode === null;
  }

  async start(): Promise<void> {
    if (this.proc && this.proc.exitCode === null) return;

    this.readyPromise = new Promise((resolve) => {
      this.readyResolve = resolve;
    });

    const pythonPath = "/usr/bin/python3";
    const scriptPath = "/home/josh/NeuroGraph/neurograph_rpc.py";

    this.proc = this.spawnFn(pythonPath, [scriptPath], {
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        NEUROGRAPH_WORKSPACE_DIR: "/home/josh/NeuroGraph/data",
        PYTHONUNBUFFERED: "1",
        // Tell module hooks they're running inside the fan-out process.
        // Elmer uses this to skip loading 3GB of brain sockets (OOM prevention).
        NEUROGRAPH_FANOUT_CONTEXT: "1",
      },
    });

    if (this.proc.stderr) {
      const stderrRl = createInterface({ input: this.proc.stderr });
      stderrRl.on("line", (line: string) => {
        this.logger.info(`[py] ${line}`);
      });
    }

    if (this.proc.stdout) {
      this.rl = createInterface({ input: this.proc.stdout });
      this.rl.on("line", (line: string) => {
        this.handleLine(line);
      });
    }

    this.proc.on("exit", (code: number | null) => {
      this.logger.warn(`Python process exited with code ${code}`);
      this.ready = false;
      this.proc = null;
      for (const [id, req] of this.pending) {
        if (req.timer) clearTimeout(req.timer);
        req.reject(new Error(`Python process exited (code ${code})`));
      }
      this.pending.clear();
    });

    await this.readyPromise;
    this.logger.info("Python RPC bridge ready");
  }

  private handleLine(line: string): void {
    let msg: JsonRpcResponse;
    try {
      msg = JSON.parse(line);
    } catch {
      this.logger.warn(`Unparseable RPC response: ${line.slice(0, 200)}`);
      return;
    }

    if ((msg as any).method === "ready") {
      this.ready = true;
      if (this.readyResolve) {
        this.readyResolve();
        this.readyResolve = null;
      }
      return;
    }

    if (msg.id == null) return;
    const req = this.pending.get(msg.id);
    if (!req) return;

    this.pending.delete(msg.id);
    if (req.timer) clearTimeout(req.timer);

    if (msg.error) {
      req.reject(new Error(`RPC error: ${msg.error.message}`));
    } else {
      req.resolve(msg.result);
    }
  }

  async ensureRunning(): Promise<void> {
    // Step 2: prohibit ensureRunning() (respawn) while stopping.
    if (this.stopping) {
      throw new Error("NeuroGraph RPC client is stopping — ensureRunning() is prohibited");
    }
    if (this.proc && this.proc.exitCode === null) return;
    this.proc = null;
    this.ready = false;
    this.logger.warn("Python process not running — restarting");
    await this.start();
    await this.call("bootstrap", {}, 60000);
  }

  async call(method: string, params: Record<string, unknown> = {}, timeoutMs = 30000): Promise<unknown> {
    // Step 1: atomically mark normal calls closed once shutdown begins.
    if (this.stopping) {
      throw new Error(`NeuroGraph RPC client is stopping — normal calls are closed (method=${method})`);
    }
    if (!this.proc || !this.proc.stdin || this.proc.exitCode !== null) {
      await this.ensureRunning();
    }

    const id = this.nextId++;
    const request = JSON.stringify({
      jsonrpc: "2.0",
      id,
      method,
      params,
    });

    const promise = new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`RPC timeout: ${method} (${timeoutMs}ms)`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });
      this.proc!.stdin!.write(request + "\n");
    });

    // Track this call as "admitted" so shutdown can drain it (step 3)
    // before issuing the privileged dispose (step 4). The tracking chain
    // is separate from the promise returned to the caller so caller-side
    // rejection semantics are unaffected.
    this.admitted.add(promise);
    promise.finally(() => this.admitted.delete(promise)).catch(() => {
      // Swallow here only to avoid an unhandled-rejection warning on the
      // *tracking* chain; the original `promise` returned above still
      // rejects normally for the caller.
    });

    return promise;
  }

  /**
   * Step 3: wait for already-admitted normal calls to settle. This is
   * deliberately unbounded — a hung call leaves shutdown pending
   * indefinitely by design (spec: "never convert the wait into an
   * automatic child signal"). Emits one diagnostic once the configured
   * warning threshold is crossed, then repeats only at a bounded cadence,
   * so a stuck shutdown is observable without ever escalating on its own.
   * Breaking an indefinitely-hung drain requires an explicit human
   * emergency action outside this client (e.g. manual process
   * intervention) — this method never performs one itself.
   */
  private async drainAdmittedCalls(): Promise<void> {
    if (this.admitted.size === 0) return;

    const warnMs = Number(process.env.NEUROGRAPH_SHUTDOWN_WARN_MS ?? 5000);
    const intervalMs = Number(process.env.NEUROGRAPH_SHUTDOWN_WARN_INTERVAL_MS ?? 30000);
    const start = Date.now();

    let repeatTimer: ReturnType<typeof setInterval> | null = null;
    const emitWarning = () => {
      if (this.admitted.size === 0) return;
      const elapsed = Date.now() - start;
      this.logger.warn(
        `NeuroGraph shutdown: ${this.admitted.size} in-flight RPC call(s) still draining after ${elapsed}ms — ` +
          `waiting indefinitely by design; will not auto-signal the child. ` +
          `Requires an explicit human emergency action to break.`
      );
    };

    const firstWarnTimer = setTimeout(() => {
      emitWarning();
      repeatTimer = setInterval(emitWarning, intervalMs);
      (repeatTimer as any).unref?.();
    }, warnMs);
    (firstWarnTimer as any).unref?.();

    try {
      await Promise.allSettled([...this.admitted]);
    } finally {
      clearTimeout(firstWarnTimer);
      if (repeatTimer) clearInterval(repeatTimer);
    }
  }

  /**
   * Step 4/5: issue exactly one privileged `dispose` request on the
   * existing Node-owned stdin stream, with no request timeout (this is a
   * lifecycle call, not a normal RPC — the old 10s/30s timeouts do not
   * apply). Memoized so repeated stop() calls never produce a second
   * dispose write.
   */
  private privilegedDispose(): Promise<DisposeResult | undefined> {
    if (this.disposePromise) return this.disposePromise;
    this.disposePromise = this.sendDisposeRequest();
    return this.disposePromise;
  }

  private sendDisposeRequest(): Promise<DisposeResult | undefined> {
    if (!this.proc || !this.proc.stdin || this.proc.exitCode !== null) {
      return Promise.reject(new Error("NeuroGraph dispose: no live process/stdin to dispose"));
    }

    const id = this.nextId++;
    const request = JSON.stringify({ jsonrpc: "2.0", id, method: "dispose", params: {} });

    return new Promise<DisposeResult | undefined>((resolve, reject) => {
      // No timer: the privileged dispose call is unbounded (spec step 5).
      this.pending.set(id, {
        resolve: (value: unknown) => resolve(value as DisposeResult | undefined),
        reject,
      });
      this.proc!.stdin!.write(request + "\n");
    });
  }

  /** Step 7: resolves on the child's natural `exit` event. */
  private waitForExit(): Promise<void> {
    if (!this.proc || this.proc.exitCode !== null) return Promise.resolve();
    return new Promise((resolve) => {
      this.proc!.once("exit", () => resolve());
    });
  }

  /**
   * Full stop lifecycle (steps 1–8). Memoized: every caller (each
   * registerService.stop() callback, however many times register()
   * appended one) converges on the same in-flight/settled promise, one
   * child, and one dispose write.
   *
   * Never sends SIGTERM/SIGKILL. Never closes stdin unless Python's
   * dispose response reports `safe_to_terminate: true`. On any failure
   * (no live process, dispose rejects, or safe_to_terminate is not true)
   * this resolves `{ closed: false }` and leaves the process, stdin, and
   * global state untouched — per spec step 8 and the "never permitted"
   * list, an unaccepted/failed shutdown must never signal the child.
   */
  async stop(): Promise<{ closed: boolean }> {
    if (this.stopPromise) return this.stopPromise;
    this.stopPromise = this.doStop();
    return this.stopPromise;
  }

  private async doStop(): Promise<{ closed: boolean }> {
    // Step 1 + 2: close admission, stop the watchdog synchronously before
    // any await so no further respawn can be scheduled once shutdown starts.
    this.stopping = true;
    this.stopWatchdog();

    // Step 3: drain already-admitted calls (indefinite wait by design).
    await this.drainAdmittedCalls();

    if (!this.proc || this.proc.exitCode !== null) {
      // Nothing to dispose — process already gone.
      if (this.rl) {
        this.rl.close();
        this.rl = null;
      }
      return { closed: true };
    }

    let disposeResult: DisposeResult | undefined;
    try {
      // Step 4/5: exactly one privileged dispose request, no timeout.
      disposeResult = await this.privilegedDispose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.logger.error(
        `NeuroGraph shutdown: dispose request failed — leaving process running, stdin open, no signal sent: ${msg}`
      );
      return { closed: false };
    }

    if (!disposeResult || disposeResult.safe_to_terminate !== true) {
      // Step 6 (fail-closed direction) + step 8: an unaccepted/absent
      // safe_to_terminate never closes stdin, never signals, never clears
      // state.
      this.logger.warn(
        "NeuroGraph shutdown: dispose response did not report safe_to_terminate=true — " +
          "leaving stdin open and process running; no signal sent."
      );
      return { closed: false };
    }

    // Step 6: close stdin only now that Python has reported readiness.
    if (this.proc.stdin) {
      this.proc.stdin.end();
    }

    // Step 7: wait for natural child exit before reporting closed.
    await this.waitForExit();

    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }

    return { closed: true };
  }
}

// ── ContextEngine Implementation ────────────────────────────────────

class NeurographContextEngine implements ContextEngine {
  readonly info: ContextEngineInfo = {
    id: "neurograph",
    name: "NeuroGraph Cognitive Substrate",
    version: "1.0.0",
    ownsCompaction: true,
  };

  private rpc: NeurographRpcClient;
  private bootstrapped = false;
  private bootstrapping: Promise<void> | null = null;
  private lastIngestedMessage: unknown = null;

  constructor(rpc: NeurographRpcClient) {
    this.rpc = rpc;
    rpc.onRespawn = () => { this.bootstrapped = false; };
  }

  /**
   * Ensure the substrate is bootstrapped before any lifecycle call.
   * OpenClaw 2026.3.13 does not guarantee bootstrap() is called before
   * afterTurn/ingest/assemble — so we self-bootstrap on first use.
   */
  private async ensureBootstrapped(): Promise<void> {
    if (this.bootstrapped) return;
    if (this.bootstrapping) {
      await this.bootstrapping;
      return;
    }
    this.rpc["logger"].info("Auto-bootstrapping NeuroGraph (OpenClaw did not call bootstrap)");
    this.bootstrapping = (async () => {
      try {
        const result = (await this.rpc.call("bootstrap", {
          sessionId: "auto",
        }, 60000)) as Record<string, unknown>;
        this.bootstrapped = true;
        this.rpc["logger"].info(
          `Auto-bootstrap succeeded — ${result?.nodes} nodes, hooks: ${result?.module_hooks}`
        );
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        this.rpc["logger"].error(`Auto-bootstrap FAILED: ${msg}`);
      } finally {
        this.bootstrapping = null;
      }
    })();
    await this.bootstrapping;
  }

  async bootstrap(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
  }): Promise<BootstrapResult> {
    this.rpc["logger"].info(`bootstrap called — session=${params.sessionId}`);
    try {
      const result = (await this.rpc.call("bootstrap", {
        sessionId: params.sessionId,
      }, 60000)) as Record<string, unknown>;
      this.bootstrapped = true;
      this.rpc["logger"].info(`bootstrap succeeded — ${result?.nodes} nodes, ${result?.module_hooks} hooks`);
      return {
        bootstrapped: true,
        reason: `NeuroGraph: ${result?.nodes ?? 0} nodes, ${result?.synapses ?? 0} synapses`,
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.rpc["logger"].error(`bootstrap FAILED: ${msg}`);
      return { bootstrapped: false, reason: `NeuroGraph bootstrap failed: ${msg}` };
    }
  }

  async ingest(params: {
    sessionId: string;
    sessionKey?: string;
    message: unknown;
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    if (params.isHeartbeat) {
      return { ingested: false };
    }

    await this.ensureBootstrapped();
    if (!this.bootstrapped) {
      return { ingested: false };
    }

    try {
      this.lastIngestedMessage = params.message;
      const result = (await this.rpc.call("ingest", {
        message: params.message,
      })) as Record<string, unknown>;
      return { ingested: result?.ingested === true };
    } catch {
      return { ingested: false };
    }
  }

  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages: unknown[];
    tokenBudget?: number;
  }): Promise<AssembleResult> {
    const messages = params.messages as any[];

    let charCount = 0;
    for (const msg of messages) {
      const content = msg?.content;
      if (typeof content === "string") {
        charCount += content.length;
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (typeof part === "string") charCount += part.length;
          else if (part?.text) charCount += String(part.text).length;
        }
      }
    }
    const estimatedTokens = Math.ceil(charCount / 4);

    await this.ensureBootstrapped();
    if (!this.bootstrapped) {
      return { messages, estimatedTokens };
    }

    try {
      const result = (await this.rpc.call("assemble", {
        messages,
      }, 15000)) as Record<string, unknown>;

      const addition = result?.systemPromptAddition;

      // KISS plumb-through (#152): if the Python side returned a truncated
      // messages array, use it so OC's replaceMessages fires and the model
      // sees the compressed context.  Guards:
      //   - Array.isArray: defend against missing/malformed response
      //   - length > 0: empty array would be pathological; fall back
      //   - Same-reference fallback (params.messages): warmup / unchanged
      //     turns preserve identity so OC's reference-equality check
      //     (`assembled.messages !== activeSession.messages`) remains a
      //     no-op.  Only turns where Python actually truncates result in
      //     a different reference → replaceMessages fires → model sees the
      //     compressed context.
      const resultMessages = result?.messages;
      const assembledMessages =
        Array.isArray(resultMessages) && resultMessages.length > 0
          ? (resultMessages as any[])
          : messages;

      return {
        messages: assembledMessages,
        estimatedTokens,
        systemPromptAddition: typeof addition === "string" ? addition : undefined,
      };
    } catch {
      return { messages, estimatedTokens };
    }
  }

  async afterTurn(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    messages: unknown[];
    prePromptMessageCount: number;
    autoCompactionSummary?: string;
    isHeartbeat?: boolean;
    tokenBudget?: number;
    runtimeContext?: Record<string, unknown>;
  }): Promise<void> {
    if (params.isHeartbeat) return;
    await this.ensureBootstrapped();
    if (!this.bootstrapped) return;

    try {
      // Extract last user message from the conversation history.
      // OpenClaw may not call ingest() so lastIngestedMessage can be null —
      // fall back to the most recent user message from the messages array.
      let lastMessage = this.lastIngestedMessage;
      if (!lastMessage && params.messages && params.messages.length > 0) {
        for (let i = params.messages.length - 1; i >= 0; i--) {
          const m = params.messages[i] as any;
          if (m?.role === "user") {
            lastMessage = m;
            break;
          }
        }
      }

      // Extract last assistant message (Syl's response this turn).
      // Punchlist #56: surfacing outcome deposit needs the full turn
      // triad — surfaced nodes + user input + Syl's response.
      let lastAssistantMessage = null;
      if (params.messages && params.messages.length > 0) {
        for (let i = params.messages.length - 1; i >= 0; i--) {
          const m = params.messages[i] as any;
          if (m?.role === "assistant") {
            lastAssistantMessage = m;
            break;
          }
        }
      }

      this.rpc["logger"].info(
        `afterTurn: sending RPC (lastUserMessage=${lastMessage ? "present" : "null"}, lastAssistantMessage=${lastAssistantMessage ? "present" : "null"}, msgs=${params.messages?.length ?? 0})`
      );
      await this.rpc.call("afterTurn", {
        lastUserMessage: lastMessage ?? undefined,
        lastAssistantMessage: lastAssistantMessage ?? undefined,
      }, 30000);
      this.rpc["logger"].info("afterTurn: RPC completed");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      this.rpc["logger"].error(`afterTurn RPC failed: ${msg}`);
    }
  }

  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    customInstructions?: string;
    currentTokenCount?: number;
    runtimeContext?: Record<string, unknown>;
  }): Promise<CompactResult> {
    if (!this.bootstrapped) {
      return { ok: true, compacted: false, reason: "not bootstrapped" };
    }

    try {
      const result = (await this.rpc.call("compact", {
        sessionId: params.sessionId,
        sessionFile: params.sessionFile,
        tokenBudget: params.tokenBudget,
        force: params.force,
        customInstructions: params.customInstructions,
      }, 60000)) as Record<string, unknown>;

      return {
        ok: result?.ok === true,
        compacted: result?.compacted === true,
        reason: typeof result?.reason === "string" ? result.reason : undefined,
        result: result?.result ? {
          summary: String((result.result as any)?.summary ?? ""),
          tokensBefore: (result.result as any)?.tokensBefore,
          tokensAfter: (result.result as any)?.tokensAfter,
          firstKeptEntryId: String((result.result as any)?.firstKeptEntryId ?? ""),
        } : undefined,
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      return { ok: false, compacted: false, reason: `compact RPC failed: ${msg}` };
    }
  }

  async dispose(): Promise<void> {
    // No-op: Python child process persists across conversation turns.
    // OpenClaw calls dispose() after each run — killing the process
    // here would destroy the substrate for all subsequent messages.
    // Actual cleanup happens when the gateway process exits.
  }
}

// ── Global shutdown-promise key ─────────────────────────────────────
// Survives across register() calls, same rationale as GLOBAL_KEY. All
// registerService("neurograph-rpc-host").stop() callbacks — whether from
// two distinct registries or repeated registration into one registry —
// call stopNeurographService(), which converges every caller on this one
// promise, so there is exactly one child, one shutdown sequence, and one
// dispose write regardless of how many times register() ran.

const SHUTDOWN_KEY = Symbol.for("neurograph.shutdownPromise");

function getGlobalShutdownPromise(): Promise<void> | undefined {
  return (globalThis as any)[SHUTDOWN_KEY];
}

function setGlobalShutdownPromise(p: Promise<void>): void {
  (globalThis as any)[SHUTDOWN_KEY] = p;
}

/**
 * Resolves the current GLOBAL_KEY state and joins the one global shutdown
 * promise (register() → registerService("neurograph-rpc-host").stop()).
 * Safe to call zero, one, or many times: the first call performs the
 * shutdown; every later call — from any registry, any number of
 * registrations — awaits the same in-flight/settled promise. Clears
 * GLOBAL_KEY only once NeurographRpcClient.stop() reports the child
 * actually closed (natural exit after an accepted safe_to_terminate);
 * a failed/unaccepted shutdown leaves global state intact so the
 * substrate keeps running rather than being silently orphaned.
 */
function stopNeurographService(): Promise<void> {
  const existingShutdown = getGlobalShutdownPromise();
  if (existingShutdown) return existingShutdown;

  const state = getGlobalState();
  if (!state || !state.rpc) {
    const noop = Promise.resolve();
    setGlobalShutdownPromise(noop);
    return noop;
  }

  const shutdown = state.rpc.stop().then((result) => {
    if (result.closed) {
      (globalThis as any)[GLOBAL_KEY] = undefined;
    }
  });

  setGlobalShutdownPromise(shutdown);
  return shutdown;
}

// ── Plugin Registration ─────────────────────────────────────────────

const neurographPlugin = {
  id: "neurograph",
  name: "NeuroGraph Cognitive Substrate",
  description:
    "Spiking neural network substrate for persistent cognitive memory. " +
    "STDP-based causal learning, semantic recall, spreading activation, " +
    "and cross-session continuity.",
  version: "1.0.0",
  kind: "context-engine" as const,

  register(api: OpenClawPluginApi) {
    const logger = api.logger;

    // Register the process-lifecycle service before any early return below,
    // per spec: every register() invocation — first, reused, or duplicate —
    // must register "neurograph-rpc-host" so OpenClaw's gateway-close path
    // has a source-owned stop() to await. start() is a no-op: registration
    // must never spawn or awaken the substrate (that stays eager-spawn /
    // on-first-use below, unchanged). stop() converges on the one global
    // shutdown promise no matter how many times this callback fires.
    api.registerService({
      id: "neurograph-rpc-host",
      start: async () => {
        // Intentional no-op — see comment above.
      },
      stop: async () => {
        await stopNeurographService();
      },
    });

    // Check if a previous register() call already spawned the organism.
    // OpenClaw calls register() in both [gateway] and [plugins] contexts
    // within the same Node.js process. Without this guard, each call
    // spawns a new Python child that fights the first for topology ownership.
    const existing = getGlobalState();
    if (existing && (existing.spawning || (existing.rpc?.isAlive?.() === true))) {
      if (existing.spawning) {
        // Spawn in progress — register a wait-factory so this context also
        // gets an engine reference once Python is ready.
        logger.info("NeuroGraph: register() during spawn — registering wait-factory");
        api.registerContextEngine("neurograph", async () => {
          for (let i = 0; i < 600; i++) {
            const s = getGlobalState();
            if (s && !s.spawning && s.rpc?.isAlive?.()) break;
            await new Promise(r => setTimeout(r, 100));
          }
          const current = getGlobalState();
          return current!.engine;
        });
      } else {
        // Process already alive — do NOT call registerContextEngine() again.
        // A second registration causes OC to dispose the first binding, which
        // closes stdin to the Python child and breaks the RPC bridge.
        // The first registration is still active and fully functional.
        logger.info(
          "NeuroGraph: register() called again, process alive — skipping re-registration to preserve stdin"
        );
      }
      return;
    }

    // First register() call — set up the singleton.
    api.registerContextEngine("neurograph", async () => {
      // Wait for eager spawn to finish if in progress
      const state = getGlobalState();
      if (state?.spawning) {
        for (let i = 0; i < 300; i++) {
          const s = getGlobalState();
          if (s && !s.spawning && s.rpc?.isAlive?.()) break;
          await new Promise(r => setTimeout(r, 100));
        }
      }

      const current = getGlobalState();
      if (current?.rpc?.isAlive?.()) {
        return current.engine;
      }

      // Eager spawn failed or hasn't run — start now
      const rpc = new NeurographRpcClient(logger);
      await rpc.start();
      const engine = new NeurographContextEngine(rpc);
      setGlobalState({ rpc, engine, spawning: false });
      return engine;
    });

    // Eager spawn — start Python process immediately.
    // The factory callback above waits for this if called during startup.
    setGlobalState({ rpc: null as any, engine: null as any, spawning: true });

    (async () => {
      try {
        const rpc = new NeurographRpcClient(logger);
        await rpc.start();
        const engine = new NeurographContextEngine(rpc);
        setGlobalState({ rpc, engine, spawning: false });
        rpc.startWatchdog();
        logger.info("NeuroGraph: eager spawn complete — organism alive (watchdog started)");
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(`NeuroGraph eager spawn failed (will retry on first use): ${msg}`);
        // Clear the spawning flag so the factory can try
        (globalThis as any)[GLOBAL_KEY] = undefined;
      }
    })();

    logger.info("NeuroGraph ContextEngine plugin registered");
  },
};

export default neurographPlugin;

// ── Test-only named exports ─────────────────────────────────────────
// The default export is what OpenClaw's extension loader consumes; these
// named exports let index.test.ts exercise the lifecycle internals (fake
// registries, fake child processes) without reaching into module
// internals via any-casts. Not part of the OpenClaw-facing contract.
export {
  NeurographRpcClient,
  NeurographContextEngine,
  getGlobalState,
  setGlobalState,
  stopNeurographService,
  getGlobalShutdownPromise,
  GLOBAL_KEY,
  SHUTDOWN_KEY,
};

