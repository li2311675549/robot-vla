import os
from robot_exam.data import shared_models as module


def test_identical_models_share_storage_while_other_models_stay_distinct(tmp_path):
    paths=[]
    for i,payload in enumerate([b'model-one',b'model-one',b'model-two']):
        p=tmp_path/f'episode{i}'/'model.mjb';p.parent.mkdir();p.write_bytes(payload);paths.append(p)
    rows=module.share_models(paths,tmp_path/'shared')
    assert os.path.samefile(paths[0],paths[1])
    assert not os.path.samefile(paths[0],paths[2])
    assert [p.read_bytes() for p in paths]==[b'model-one',b'model-one',b'model-two']
    assert len({r['sha256'] for r in rows})==2
    assert not paths[0].stat().st_mode&0o222
