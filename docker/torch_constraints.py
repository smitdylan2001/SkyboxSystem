"""Print pip constraints that pin the base image's torch/torchvision (local '+cuXXX' suffix dropped)."""

import importlib.metadata as metadata

for name in ("torch", "torchvision"):
    try:
        print(f"{name}=={metadata.version(name).split('+')[0]}")
    except metadata.PackageNotFoundError:
        pass
