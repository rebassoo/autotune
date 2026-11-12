"""
Copy a YAML config, setting dotted keys: python scripts/set_config_keys.py IN OUT a.b=1 c=x
Values are parsed as YAML (so 0.0 is a float, true a bool, text a string).
"""
import sys

import yaml


def main():
    src, dst, *pairs = sys.argv[1:]
    cfg = yaml.safe_load(open(src))
    for pair in pairs:
        key, val = pair.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(val)
    with open(dst, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


if __name__ == "__main__":
    main()
