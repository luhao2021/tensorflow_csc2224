import zlib
import base64

with open("find_cuda_config.py", "rb") as reader:
    code = base64.b64encode(zlib.compress(reader.read()))

with open("find_cuda_config.py.gz.base64", "wb") as writer:
    writer.write(code)
