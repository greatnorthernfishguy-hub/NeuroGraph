#!/usr/bin/env python3
"""
Unified Patch: Strength-Modulated Learning + First-Class Reasoning Strings

Combines punch list #16 (reasoning/interface) with the substrate foundation
work (strength-modulated Hebbian learning, synapse metadata accumulation).

This is the foundational change that makes the substrate ready for the triad.

== ng_lite.py (canonical, NeuroGraph repo) ==

  record_outcome():
    - New `strength: float = 1.0` parameter
    - Hebbian delta modulated: delta * strength
    - Synapse metadata accumulates strength_sum, strength_count, last_context
    - Bridge forwarding includes strength in metadata
    - Docstring documents the design intent

  get_recommendations():
    - Returns List[Tuple[str, float, str]] (was 2-tuples)
    - Bridge path: stop stripping reasoning
    - Local path: generate reasoning via _build_local_reasoning()

  _build_local_reasoning():  (NEW METHOD)
    - Single evolution point for reasoning generation
    - Reads synapse stats AND strength signatures from metadata
    - Extraction boundary — topology becomes human-legible here

== ng_ecosystem.py (canonical, NeuroGraph repo) ==

  record_outcome():
    - Accepts and passes through `strength` parameter
    - Returns {} not None when substrate unavailable
    - Return type: Dict[str, Any] not Optional

  get_recommendations():
    - Returns [] not None when substrate unavailable
    - Return type: List[Tuple[str, float, str]] not Optional

  get_context():
    - Remove dead `or []` fallback

== router.py (TID repo) ==

  _score_learned():
    - Unpack 3-tuples (target_id, weight, _reasoning)
    - Remove null-check workaround

== Vendored files ==

  TID, TrollGuard: byte-for-byte copy of canonical ng_lite.py, ng_ecosystem.py

== Tests ==

  Fix exact-tuple assertions in NeuroGraph and TID test_ng_lite.py

Usage:
    python3 ~/apply_substrate_foundation.py          # dry run
    python3 ~/apply_substrate_foundation.py --apply   # apply changes
"""

import os
import sys
from pathlib import Path

DRY_RUN = "--apply" not in sys.argv

HOME = Path(os.environ.get("PATCH_HOME", str(Path.home())))
NG_DIR = HOME / "NeuroGraph"
TID_DIR = HOME / "The-Inference-Difference"
TG_DIR = HOME / "TrollGuard"

results = []


def replace_in_file(filepath: Path, old: str, new: str, description: str):
    """Replace exactly one occurrence of old with new in filepath."""
    if not filepath.exists():
        results.append(f"  SKIP (not found): {filepath}")
        return False

    content = filepath.read_text()
    count = content.count(old)

    if count == 0:
        results.append(f"  SKIP (pattern not found): {filepath} — {description}")
        return False
    if count > 1:
        results.append(f"  ERROR ({count} matches, expected 1): {filepath} — {description}")
        return False

    if DRY_RUN:
        results.append(f"  WOULD: {description} in {filepath}")
        return True

    new_content = content.replace(old, new, 1)
    filepath.write_text(new_content)
    results.append(f"  DONE: {description} in {filepath}")
    return True


# ===========================================================================
# 1. ng_lite.py — Canonical (NeuroGraph repo)
# ===========================================================================
print("=== ng_lite.py (canonical) ===")

NG_LITE = NG_DIR / "ng_lite.py"

# 1a. record_outcome signature: add strength parameter
replace_in_file(
    NG_LITE,
    '    def record_outcome(\n'
    '        self,\n'
    '        embedding: np.ndarray,\n'
    '        target_id: str,\n'
    '        success: bool,\n'
    '        metadata: Optional[Dict[str, Any]] = None,\n'
    '    ) -> Dict[str, Any]:',
    '    def record_outcome(\n'
    '        self,\n'
    '        embedding: np.ndarray,\n'
    '        target_id: str,\n'
    '        success: bool,\n'
    '        strength: float = 1.0,\n'
    '        metadata: Optional[Dict[str, Any]] = None,\n'
    '    ) -> Dict[str, Any]:',
    "record_outcome: add strength parameter",
)

# 1b. record_outcome docstring: document strength + design intent
replace_in_file(
    NG_LITE,
    '        """Record an outcome and update learning weights.\n'
    '\n'
    '        This is the core learning method. Call it after every\n'
    '        decision to teach NG-Lite what works and what doesn\'t.\n'
    '\n'
    '        Hebbian rule:\n'
    '            - Success: weight += success_boost * (1.0 - weight)\n'
    '              (diminishing returns as weight approaches 1.0)\n'
    '            - Failure: weight -= failure_penalty * weight\n'
    '              (proportional to current confidence)\n'
    '\n'
    '        If a bridge to NeuroGraph SaaS is connected, the outcome\n'
    '        is also forwarded there for cross-module learning.\n'
    '\n'
    '        Args:\n'
    '            embedding: The input pattern embedding (1-D numpy array).\n'
    '            target_id: What was chosen (model name, action, etc.).\n'
    '            success: Whether the outcome was successful.\n'
    '            metadata: Optional context about this outcome.\n'
    '\n'
    '        Returns:\n'
    '            Dict with learning results (node_id, weight_after, etc.).\n'
    '\n'
    '        Raises:\n'
    '            ValueError: If embedding is not a 1-D numpy array.\n'
    '        """',
    '        """Record an outcome and update learning weights.\n'
    '\n'
    '        This is the core learning method. Call it after every\n'
    '        decision to teach NG-Lite what works and what doesn\'t.\n'
    '\n'
    '        Hebbian rule (strength-modulated):\n'
    '            - Success: weight += success_boost * (1 - weight) * strength\n'
    '            - Failure: weight -= failure_penalty * weight * strength\n'
    '\n'
    '        The strength parameter lets callers indicate how significant\n'
    '        this outcome was in their domain.  High-severity TrollGuard\n'
    '        detections or divergent TID quality scores teach harder than\n'
    '        routine confirmations.  Default 1.0 preserves backward compat.\n'
    '\n'
    '        Strength experience accumulates on the synapse as metadata,\n'
    '        giving the topology a record of how intensely each connection\n'
    '        was forged.  At Tier 3, NeuroGraph proper reads these\n'
    '        signatures to distinguish battle-tested synapses from routine.\n'
    '\n'
    '        If a bridge is connected, the outcome is forwarded for\n'
    '        cross-module learning with strength included in metadata.\n'
    '\n'
    '        Args:\n'
    '            embedding: The input pattern embedding (1-D numpy array).\n'
    '            target_id: What was chosen (model name, action, etc.).\n'
    '            success: Whether the outcome was successful.\n'
    '            strength: Learning intensity [0.0, 1.0].  How significant\n'
    '                this outcome was in the caller\'s domain.  Default 1.0.\n'
    '            metadata: Optional caller context.  Stored on the synapse\n'
    '                as last_context for extraction-boundary use.\n'
    '\n'
    '        Returns:\n'
    '            Dict with learning results (node_id, weight_after, etc.).\n'
    '\n'
    '        Raises:\n'
    '            ValueError: If embedding is not a 1-D numpy array.\n'
    '        """',
    "record_outcome: docstring with strength + design intent",
)

# 1c. Hebbian update: strength modulation + metadata accumulation
replace_in_file(
    NG_LITE,
    '        synapse.activation_count += 1\n'
    '\n'
    '        if success:\n'
    '            synapse.success_count += 1\n'
    '            # Hebbian strengthening with soft saturation\n'
    '            delta = self.config["success_boost"] * (1.0 - synapse.weight)\n'
    '            synapse.weight += delta\n'
    '        else:\n'
    '            synapse.failure_count += 1\n'
    '            # Anti-Hebbian weakening proportional to current weight\n'
    '            delta = self.config["failure_penalty"] * synapse.weight\n'
    '            synapse.weight -= delta\n'
    '\n'
    '        synapse.weight = float(np.clip(synapse.weight, 0.0, 1.0))\n'
    '        synapse.last_updated = time.time()',
    '        synapse.activation_count += 1\n'
    '\n'
    '        # Clamp strength to valid range\n'
    '        strength = float(np.clip(strength, 0.0, 1.0))\n'
    '\n'
    '        if success:\n'
    '            synapse.success_count += 1\n'
    '            # Hebbian strengthening, modulated by caller-reported significance\n'
    '            delta = self.config["success_boost"] * (1.0 - synapse.weight) * strength\n'
    '            synapse.weight += delta\n'
    '        else:\n'
    '            synapse.failure_count += 1\n'
    '            # Anti-Hebbian weakening, modulated by caller-reported significance\n'
    '            delta = self.config["failure_penalty"] * synapse.weight * strength\n'
    '            synapse.weight -= delta\n'
    '\n'
    '        synapse.weight = float(np.clip(synapse.weight, 0.0, 1.0))\n'
    '        synapse.last_updated = time.time()\n'
    '\n'
    '        # Accumulate strength experience on synapse —\n'
    '        # the topology remembers how intensely it was taught\n'
    '        synapse.metadata["strength_sum"] = synapse.metadata.get("strength_sum", 0.0) + strength\n'
    '        synapse.metadata["strength_count"] = synapse.metadata.get("strength_count", 0) + 1\n'
    '        if metadata:\n'
    '            synapse.metadata["last_context"] = metadata',
    "Hebbian update: strength modulation + metadata accumulation",
)

# 1d. Bridge forwarding: include strength in metadata
replace_in_file(
    NG_LITE,
    '        # Forward to bridge if connected\n'
    '        if self._bridge and self._bridge.is_connected():\n'
    '            try:\n'
    '                enriched = self._bridge.record_outcome(\n'
    '                    embedding=embedding,\n'
    '                    target_id=target_id,\n'
    '                    success=success,\n'
    '                    module_id=self.module_id,\n'
    '                    metadata=metadata,\n'
    '                )',
    '        # Forward to bridge if connected (include strength for Tier 2/3)\n'
    '        if self._bridge and self._bridge.is_connected():\n'
    '            try:\n'
    '                bridge_meta = dict(metadata or {})\n'
    '                bridge_meta["strength"] = strength\n'
    '                enriched = self._bridge.record_outcome(\n'
    '                    embedding=embedding,\n'
    '                    target_id=target_id,\n'
    '                    success=success,\n'
    '                    module_id=self.module_id,\n'
    '                    metadata=bridge_meta,\n'
    '                )',
    "Bridge forwarding: include strength in metadata",
)

# 1e. get_recommendations return type: 2-tuple → 3-tuple
replace_in_file(
    NG_LITE,
    '    ) -> List[Tuple[str, float]]:\n'
    '        """Get target recommendations for an input pattern.',
    '    ) -> List[Tuple[str, float, str]]:\n'
    '        """Get target recommendations for an input pattern.',
    "get_recommendations: return type 2-tuple → 3-tuple",
)

# 1f. get_recommendations docstring Returns section
replace_in_file(
    NG_LITE,
    '        Returns:\n'
    '            List of (target_id, confidence) tuples, highest first.\n'
    '            Empty list if no learned routes exist for this pattern.',
    '        Returns:\n'
    '            List of (target_id, confidence, reasoning) tuples, highest\n'
    '            first.  The reasoning string captures the experience behind\n'
    '            each recommendation — learning mechanism, success ratio,\n'
    '            weight, activation volume, and strength signature.\n'
    '            Empty list if no learned routes exist for this pattern.',
    "get_recommendations: docstring returns → 3-tuple",
)

# 1g. Bridge path: stop stripping reasoning
replace_in_file(
    NG_LITE,
    '                if bridge_recs:\n'
    '                    # Bridge returns (target, confidence, reasoning)\n'
    '                    # We return (target, confidence) for API consistency\n'
    '                    return [(t, c) for t, c, _ in bridge_recs]',
    '                if bridge_recs:\n'
    '                    return bridge_recs',
    "Bridge path: stop stripping reasoning",
)

# 1h. Local path: generate reasoning + new _build_local_reasoning method
replace_in_file(
    NG_LITE,
    '        # Local learning\n'
    '        node = self.find_or_create_node(embedding)\n'
    '\n'
    '        relevant = [\n'
    '            (syn.target_id, syn.weight)\n'
    '            for key, syn in self.synapses.items()\n'
    '            if key[0] == node.node_id and syn.weight > self.config["pruning_threshold"]\n'
    '        ]\n'
    '\n'
    '        if not relevant:\n'
    '            return []\n'
    '\n'
    '        relevant.sort(key=lambda x: x[1], reverse=True)\n'
    '        return relevant[:top_k]\n'
    '\n'
    '    def detect_novelty',
    '        # Local learning\n'
    '        node = self.find_or_create_node(embedding)\n'
    '\n'
    '        relevant = []\n'
    '        for key, syn in self.synapses.items():\n'
    '            if key[0] == node.node_id and syn.weight > self.config["pruning_threshold"]:\n'
    '                reasoning = self._build_local_reasoning(syn)\n'
    '                relevant.append((syn.target_id, syn.weight, reasoning))\n'
    '\n'
    '        if not relevant:\n'
    '            return []\n'
    '\n'
    '        relevant.sort(key=lambda x: x[1], reverse=True)\n'
    '        return relevant[:top_k]\n'
    '\n'
    '    def _build_local_reasoning(self, synapse: NGLiteSynapse) -> str:\n'
    '        """Generate reasoning string from local Hebbian experience.\n'
    '\n'
    '        Single point of evolution for how NG-Lite articulates its local\n'
    '        learning.  V1 renders synapse stats and strength signatures.\n'
    '        As NG-Lite gains meta-learning capability (punch list #21),\n'
    '        this method becomes the place where reasoning generation\n'
    '        itself improves.\n'
    '\n'
    '        This is an extraction boundary — topology becomes human-legible\n'
    '        here.  The substrate doesn\'t need these labels; consumers and\n'
    '        dashboards do.\n'
    '\n'
    '        Args:\n'
    '            synapse: The synapse whose experience to articulate.\n'
    '\n'
    '        Returns:\n'
    '            Human-readable reasoning grounded in actual experience data.\n'
    '        """\n'
    '        total = synapse.success_count + synapse.failure_count\n'
    '        if total > 0:\n'
    '            detail = f"w={synapse.weight:.2f}, {synapse.activation_count} activations"\n'
    '            strength_count = synapse.metadata.get("strength_count", 0)\n'
    '            if strength_count > 0:\n'
    '                avg = synapse.metadata["strength_sum"] / strength_count\n'
    '                detail += f", avg_strength={avg:.2f}"\n'
    '            return (\n'
    '                f"Hebbian: {synapse.success_count}/{total} success ({detail})"\n'
    '            )\n'
    '        return f"Hebbian: no outcomes yet (w={synapse.weight:.2f})"\n'
    '\n'
    '    def detect_novelty',
    "Local path: reasoning generation + _build_local_reasoning with strength",
)


# ===========================================================================
# 2. ng_ecosystem.py — Canonical (NeuroGraph repo)
# ===========================================================================
print("\n=== ng_ecosystem.py (canonical) ===")

NG_ECO = NG_DIR / "ng_ecosystem.py"

# 2a. record_outcome: add strength, fix return type, return {} not None
replace_in_file(
    NG_ECO,
    '    def record_outcome(\n'
    '        self,\n'
    '        embedding: np.ndarray,\n'
    '        target_id: str,\n'
    '        success: bool,\n'
    '        metadata: Optional[Dict[str, Any]] = None,\n'
    '    ) -> Optional[Dict[str, Any]]:\n'
    '        """Record a learning outcome.\n'
    '\n'
    '        The embedding is the semantic representation of the input.\n'
    '        The target_id is an opaque string representing what was decided\n'
    '        (e.g., "model:llama3", "threat:prompt_injection", "action:search").\n'
    '\n'
    '        Returns enriched response from the active bridge (Tier 2/3) or\n'
    '        None if only Tier 1 is active.\n'
    '        """\n'
    '        if self._ng is None:\n'
    '            return None\n'
    '        with self._ops_lock:\n'
    '            return self._ng.record_outcome(\n'
    '                embedding, target_id, success, metadata=metadata\n'
    '            )',
    '    def record_outcome(\n'
    '        self,\n'
    '        embedding: np.ndarray,\n'
    '        target_id: str,\n'
    '        success: bool,\n'
    '        strength: float = 1.0,\n'
    '        metadata: Optional[Dict[str, Any]] = None,\n'
    '    ) -> Dict[str, Any]:\n'
    '        """Record a learning outcome.\n'
    '\n'
    '        The embedding is the semantic representation of the input.\n'
    '        The target_id is an opaque string representing what was decided\n'
    '        (e.g., "model:llama3", "threat:prompt_injection", "action:search").\n'
    '\n'
    '        Returns the learning result dict from the substrate.\n'
    '        """\n'
    '        if self._ng is None:\n'
    '            return {}\n'
    '        with self._ops_lock:\n'
    '            return self._ng.record_outcome(\n'
    '                embedding, target_id, success, strength=strength, metadata=metadata\n'
    '            )',
    "record_outcome: add strength passthrough, return {} not None",
)

# 2b. get_recommendations: return [] not None, fix annotation
replace_in_file(
    NG_ECO,
    '    ) -> Optional[List[Tuple[str, float, str]]]:\n'
    '        """Get recommendations from the active learning substrate.\n'
    '\n'
    '        Returns list of (target_id, confidence, reasoning) or None.\n'
    '\n'
    '        At Tier 1, returns local recommendations only.\n'
    '        At Tier 2, includes cross-module peer patterns.\n'
    '        At Tier 3, includes full SNN recommendations + hyperedge context.\n'
    '        """\n'
    '        if self._ng is None:\n'
    '            return None',
    '    ) -> List[Tuple[str, float, str]]:\n'
    '        """Get recommendations from the active learning substrate.\n'
    '\n'
    '        Returns list of (target_id, confidence, reasoning).\n'
    '\n'
    '        At Tier 1, returns local recommendations only.\n'
    '        At Tier 2, includes cross-module peer patterns.\n'
    '        At Tier 3, includes full SNN recommendations + hyperedge context.\n'
    '        """\n'
    '        if self._ng is None:\n'
    '            return []',
    "get_recommendations: return [] not None, fix return type",
)

# 2c. get_context: remove dead 'or []' fallback
replace_in_file(
    NG_ECO,
    '        recs = self.get_recommendations(embedding, top_k=top_k) or []',
    '        recs = self.get_recommendations(embedding, top_k=top_k)',
    "get_context: remove dead 'or []' fallback",
)


# ===========================================================================
# 3. router.py — TID
# ===========================================================================
print("\n=== router.py (TID) ===")

ROUTER = TID_DIR / "inference_difference" / "router.py"

# 3a. Remove null-check workaround + unpack 3-tuples
replace_in_file(
    ROUTER,
    '        recs = self._ng_lite.get_recommendations(embedding, top_k=20)\n'
    '        if recs is None:\n'
    '            return 0.5  # Substrate has no opinion\n'
    '\n'
    '        for target_id, weight in recs:',
    '        recs = self._ng_lite.get_recommendations(embedding, top_k=20)\n'
    '\n'
    '        for target_id, weight, _reasoning in recs:',
    "Remove null-check workaround, unpack 3-tuples",
)


# ===========================================================================
# 4. Vendored file updates (byte-for-byte copy)
# ===========================================================================
print("\n=== Vendored file sync ===")

for filename in ["ng_lite.py", "ng_ecosystem.py"]:
    src = NG_DIR / filename
    # TID
    dst = TID_DIR / filename
    if src.exists() and dst.exists():
        if DRY_RUN:
            results.append(f"  WOULD: copy {src} → {dst}")
        else:
            dst.write_text(src.read_text())
            results.append(f"  DONE: copy {src} → {dst}")
    # TrollGuard
    dst = TG_DIR / filename
    if src.exists() and dst.exists():
        if DRY_RUN:
            results.append(f"  WOULD: copy {src} → {dst}")
        else:
            dst.write_text(src.read_text())
            results.append(f"  DONE: copy {src} → {dst}")


# ===========================================================================
# 5. Test updates
# ===========================================================================
print("\n=== Test fixes ===")

replace_in_file(
    NG_DIR / "tests" / "test_ng_lite.py",
    'assert recs[0] == ("bridge_model", 0.95)',
    'assert recs[0] == ("bridge_model", 0.95, "cross-module recommendation")',
    "Fix exact-tuple assertion for 3-tuple",
)

replace_in_file(
    TID_DIR / "tests" / "test_ng_lite.py",
    'assert recs[0] == ("bridge_model", 0.95)',
    'assert recs[0] == ("bridge_model", 0.95, "cross-module recommendation")',
    "Fix exact-tuple assertion for 3-tuple",
)


# ===========================================================================
# Report
# ===========================================================================
print("\n" + "=" * 60)
if DRY_RUN:
    print("DRY RUN — no files modified. Run with --apply to apply.")
else:
    print("ALL CHANGES APPLIED.")
print("=" * 60)
for r in results:
    print(r)

if DRY_RUN:
    print(f"\nRe-run with: python3 {sys.argv[0]} --apply")
