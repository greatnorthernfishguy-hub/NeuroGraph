import json,sys,os,subprocess
S=sys.argv[1]; PG="/home/josh/NeuroGraph-worktrees/z12-want-hub-build-20260930/handoffs/z12-want-hub-build/pg1/records/"
C=["return","removed_count","removed_ids_sha256_in_order","removed_ids_sha256_sorted","pruned_events","synapses_before","synapses_after","pre_state_digest","state_digest_before_checkpoint","state_digest","checkpoint_sha256","checkpoint_size","counts_after_restore","counts_final"]
R=lambda n: json.load(open(S+"/records/"+n)); B=lambda n: json.load(open(PG+n))
out={}
for c in ("a","b"):
    mb,mf=R("part1-%s-base.json"%c),R("part1-%s-fold.json"%c); bb,bf=B("part1-%s-base.json"%c),B("part1-%s-fold.json"%c)
    o={"base_vs_fold_mine":{k:mb[k]==mf[k] for k in C},"mine_vs_builder_base":{k:mb[k]==bb[k] for k in C},"mine_vs_builder_fold":{k:mf[k]==bf[k] for k in C},
       "own_digest_pre_equal":mb["own_digest_pre"]==mf["own_digest_pre"],"own_digest_post_equal":mb["own_digest_post"]==mf["own_digest_post"],
       "own_digest_post_sha256":mb["own_digest_post"]["sha256"],"own_digest_pre_sha256":mb["own_digest_pre"]["sha256"],"own_digest_fields":mb["own_digest_post"]["fields"],
       "own_digest_n":[mb["own_digest_pre"]["n_synapses_hashed"],mb["own_digest_post"]["n_synapses_hashed"]],
       "own_digest_pre_neq_post":mb["own_digest_pre"]["sha256"]!=mb["own_digest_post"]["sha256"],
       "header":{k:[m["header"]["neuro_foundation_file"],m["header"]["git_rev"],m["header"]["blob"],m["header"]["new_api_present"],m["header"]["void"],m["header"]["ng_tract_file"],m["header"]["ng_tract_version"],m["header"]["pythonhashseed"],m["header"]["env"]] for k,m in (("base",mb),("fold",mf))},
       "main_sha":[mb["main_sha256_before"],mb["main_sha256_after"],mf["main_sha256_before"],mf["main_sha256_after"]],"main_equal":[mb["main_equal_before_after"],mf["main_equal_before_after"]],"main_stat_equal":[mb["main_stat_equal"],mf["main_stat_equal"]],
       "audit":{k:[m["audit"]["write_mode_opens"],m["audit"]["mutating_calls"],m["audit"]["spawned_programs"],m["audit"]["data_read_opens"]] for k,m in (("base",mb),("fold",mf))},
       "memory":{k:[m["ru_maxrss_after_restore_gib"],m["memory"]["ru_maxrss_gib"],m["memory"]["cgroup_memory_peak_gib"],m["memory"]["cgroup_memory_events"],m["cgroup"]["memory.max"],m["cgroup"]["memory.swap.max"]] for k,m in (("base",mb),("fold",mf))},
       "mine":{k:mb[k] for k in C},"removed_is_sorted_order":mb["removed_is_sorted_order"],"removed_unique":mb["removed_ids_unique"],"prune_wall":[mb["prune_wall_s"],mf["prune_wall_s"]]}
    out[c]=o
print(json.dumps(out,indent=1,sort_keys=True))
