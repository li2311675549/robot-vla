"""The non-seekable upload body must be a readable ZIP with an exact length."""
import importlib.util
import io
from pathlib import Path
import zipfile


def test_release_zip_stream_length_and_payload(tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/upload_github_release_stream.py'
    spec=importlib.util.spec_from_file_location('release_stream',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    text=tmp_path/'指标.md';text.write_text('真实物理记录\n')
    binary=tmp_path/'binary.bin';binary.write_bytes(bytes(range(256))*100)
    files=[(text,'资料/指标.md'),(binary,'资料/binary.bin')]
    count=module.CountSink();module.write_zip(count,files)
    class Connection:
        def __init__(self):self.parts=[]
        def send(self,data):self.parts.append(bytes(data))
    connection=Connection();sink=module.NetworkSink(connection,count.count,'local-test')
    module.write_zip(sink,files);payload=b''.join(connection.parts)
    assert len(payload)==count.count==sink.count
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        assert archive.read('资料/指标.md')==text.read_bytes()
        assert archive.read('资料/binary.bin')==binary.read_bytes()
