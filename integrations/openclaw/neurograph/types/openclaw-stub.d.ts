/**
 * Ambient stand-in for the `openclaw` npm package's plugin-SDK type surface.
 *
 * Changelog:
 * [2026-09-13] Claude Sonnet 5 — Source Lifecycle Repair, chunk 1
 *   What: Declares `openclaw/plugin-sdk/plugins/types.js` (OpenClawPluginApi)
 *         and `openclaw/plugin-sdk/context-engine/types.js` (ContextEngine
 *         and its lifecycle result types), scoped to exactly the members
 *         integrations/openclaw/neurograph/index.ts imports and uses.
 *   Why:  `openclaw` is an optional peerDependency (see package.json) and is
 *         not installed in this dev/test environment — there is no VPS
 *         OpenClaw checkout available to typecheck against here. This stub
 *         lets `tsc --noEmit` run at all in that environment; a real
 *         OpenClaw install at deploy time supersedes this stub via normal
 *         module resolution (this file only declares ambient modules, it
 *         does not shadow a real `openclaw` package if one is present in
 *         node_modules). See the 2026-09-13 follow-up entry below: a clean
 *         run here does NOT by itself satisfy the spec's "type-checks
 *         against the installed OpenClaw SDK" requirement — that gate is
 *         separately tracked as open.
 *   How:  Ambient `declare module` blocks matching the exact import sites in
 *         index.ts (verified via grep — no other openclaw import path or
 *         API-surface member is referenced by this chunk's code).
 *
 * [2026-09-13] Claude Sonnet 5 — corrective follow-up (Grok family review of 0ac8826)
 *   What: Clarified scope: this stub is an isolated compile harness only.
 *   Why:  A clean `npm run typecheck:stub-harness` against this file proves
 *         index.ts is internally type-consistent with these declared shapes —
 *         it does NOT satisfy the frozen "type-checks against the installed
 *         OpenClaw SDK" gate. This laptop has no installed `openclaw`
 *         package to check against, and the real runtime SDK lives on the
 *         VPS, which this workstream is forbidden from reaching (no SSH or
 *         live inspection). That gate is explicitly left open, pending a
 *         later controlled VPS validation pass against the real package.
 *         Do not treat a pass here as SDK compatibility proof.
 *   How:  Header comment only; no declared shapes changed.
 *
 * [2026-09-13] Claude Sonnet 5 — second corrective follow-up (Grok second-pass review)
 *   What: Corrected the first (chunk-1) paragraph above, which still claimed
 *         a clean run here "satisfies" the installed-SDK test requirement —
 *         that claim was already superseded by the entry directly below it
 *         but the original wording was never fixed, leaving the file
 *         self-contradictory.
 *   Why:  Independent second-pass review flagged the stale claim. The real-
 *         SDK/activation gate stays explicitly open; this stub proves only
 *         internal type-consistency against its own declared shapes.
 *   How:  Wording fix only; no declared shapes changed.
 */

declare module "openclaw/plugin-sdk/plugins/types.js" {
  export type PluginLogger = {
    info: (msg: string) => void;
    error: (msg: string) => void;
    warn: (msg: string) => void;
  };

  export type PluginServiceDefinition = {
    id: string;
    start: () => Promise<void>;
    stop: () => Promise<void>;
  };

  export interface OpenClawPluginApi {
    logger: PluginLogger;
    registerService(service: PluginServiceDefinition): void;
    registerContextEngine(
      id: string,
      factory: () => Promise<import("openclaw/plugin-sdk/context-engine/types.js").ContextEngine>
    ): void;
  }
}

declare module "openclaw/plugin-sdk/context-engine/types.js" {
  export type ContextEngineInfo = {
    id: string;
    name: string;
    version: string;
    ownsCompaction?: boolean;
  };

  export type BootstrapResult = {
    bootstrapped: boolean;
    reason?: string;
  };

  export type IngestResult = {
    ingested: boolean;
  };

  export type AssembleResult = {
    messages: unknown[];
    estimatedTokens?: number;
    systemPromptAddition?: string;
  };

  export type CompactResult = {
    ok: boolean;
    compacted: boolean;
    reason?: string;
    result?: {
      summary: string;
      tokensBefore?: number;
      tokensAfter?: number;
      firstKeptEntryId: string;
    };
  };

  export interface ContextEngine {
    readonly info: ContextEngineInfo;
    bootstrap(params: {
      sessionId: string;
      sessionKey?: string;
      sessionFile: string;
    }): Promise<BootstrapResult>;
    ingest(params: {
      sessionId: string;
      sessionKey?: string;
      message: unknown;
      isHeartbeat?: boolean;
    }): Promise<IngestResult>;
    assemble(params: {
      sessionId: string;
      sessionKey?: string;
      messages: unknown[];
      tokenBudget?: number;
    }): Promise<AssembleResult>;
    afterTurn(params: {
      sessionId: string;
      sessionKey?: string;
      sessionFile: string;
      messages: unknown[];
      prePromptMessageCount: number;
      autoCompactionSummary?: string;
      isHeartbeat?: boolean;
      tokenBudget?: number;
      runtimeContext?: Record<string, unknown>;
    }): Promise<void>;
    compact(params: {
      sessionId: string;
      sessionKey?: string;
      sessionFile: string;
      tokenBudget?: number;
      force?: boolean;
      customInstructions?: string;
      currentTokenCount?: number;
      runtimeContext?: Record<string, unknown>;
    }): Promise<CompactResult>;
    dispose(): Promise<void>;
  }
}
