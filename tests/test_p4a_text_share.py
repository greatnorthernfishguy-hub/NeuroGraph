"""P4a (2026-10-05): restore shares identical large metadata texts; values, equality and checkpoint bytes unchanged."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neuro_foundation as nf  # noqa: E402

BIG = 'turn text ' * 100          # 1,000 chars, over the pool threshold
SMALL = 'tiny'


def _graph(tmp_path):
    g = nf.Graph()
    for i in range(6):
        g.create_node(node_id=f'n{i}', metadata={'_forest_content': ''.join(['turn text '] * 100), 'kind': SMALL,
                                                 'other': f'unique {i} ' * 40})
    p = str(tmp_path / 'g.msgpack')
    g.checkpoint(p)
    return g, p


def test_restore_shares_identical_large_texts_and_keeps_values(tmp_path):
    g, p = _graph(tmp_path)
    h = nf.Graph(); h.restore(p)
    texts = [h.nodes[f'n{i}'].metadata['_forest_content'] for i in range(6)]
    assert all(t == BIG for t in texts)
    assert len({id(t) for t in texts}) == 1                  # one shared object
    others = [h.nodes[f'n{i}'].metadata['other'] for i in range(6)]
    assert len({id(t) for t in others}) == 6                 # distinct texts stay distinct
    assert h.nodes['n0'].metadata['kind'] == SMALL


def test_checkpoint_bytes_unchanged_by_sharing(tmp_path):
    g, p = _graph(tmp_path)
    h = nf.Graph(); h.restore(p)
    p2 = str(tmp_path / 'g2.msgpack'); h.checkpoint(p2)
    k = nf.Graph(); k.restore(p2)
    p3 = str(tmp_path / 'g3.msgpack'); k.checkpoint(p3)
    assert open(p2, 'rb').read() == open(p3, 'rb').read()


def test_helper_leaves_non_dicts_and_small_values_alone():
    pool = {}
    assert nf._share_metadata_texts(None, pool) is None
    m = {'a': SMALL, 'b': 5, 'c': BIG}
    assert nf._share_metadata_texts(m, pool) is m and m['a'] is SMALL and m['c'] == BIG and pool == {BIG: BIG}
