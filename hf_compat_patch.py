"""
Compatibility patch for sentence-transformers 2.2.0 with huggingface_hub 0.36.2
Monkey-patches the renamed function so old code works.
"""
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'cached_download'):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass  # Fail silently if huggingface_hub not installed
