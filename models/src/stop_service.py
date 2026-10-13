"""Request a clean shutdown from one local tool service."""

import argparse
import json
from urllib.request import ProxyHandler, Request, build_opener

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
request = Request(
    f"http://127.0.0.1:{args.port}/shutdown", data=b"{}",
    headers={"Content-Type": "application/json"}, method="POST",
)
with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
    print(json.dumps(json.load(response), indent=2))
