"""
Expand configs/sweep.yaml into one JSON config per run.

One config -> one result file, so a run is identified by its config and nothing
else. Axes vary one key at a time from `base`; the base config itself is emitted
once. Duplicates across axes (e.g. fm=virchow2 appears in `encoder` and equals the
base) are de-duplicated by run_id, so each unique setting is run exactly once.
"""

import argparse
import hashlib
import itertools
import json
import os

import yaml

SHORT = {
    'fm': {'features_': ''},
}


def slug(cfg, varied):
    """Readable, deterministic id from the keys that differ from base."""
    parts = []
    for k in sorted(varied):
        v = cfg[k]
        if k == 'fm':
            v = str(v).replace('features_', '')
        elif isinstance(v, (list, tuple)):
            v = f"{len(v)}fam"
        parts.append(f"{k}-{v}")
    return "__".join(parts) if parts else "base"


def expand(spec):
    base = dict(spec['base'])
    out = {}

    def add(cfg, axis, varied):
        cfg = dict(cfg)
        rid = slug(cfg, varied)
        cfg['run_id'] = rid
        cfg['axis'] = axis
        # Stable hash of the semantic content, for provenance.
        payload = {k: v for k, v in cfg.items() if k not in ('run_id', 'axis')}
        cfg['config_hash'] = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
        out.setdefault(rid, cfg)

    add(base, 'base', [])

    for axis, keys in spec.get('axes', {}).items():
        names = list(keys)
        for combo in itertools.product(*(keys[n] for n in names)):
            cfg = dict(base)
            cfg.update(dict(zip(names, combo)))
            varied = [n for n in names if cfg[n] != base.get(n)] or names
            add(cfg, axis, varied)

    # Crosses vary several keys jointly; always name every crossed key in the id
    # so interaction cells never collide with main-effect cells.
    for axis, keys in spec.get('crosses', {}).items():
        names = list(keys)
        for combo in itertools.product(*(keys[n] for n in names)):
            cfg = dict(base)
            cfg.update(dict(zip(names, combo)))
            add(cfg, axis, names)

    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spec', default='configs/sweep.yaml')
    ap.add_argument('--out_dir', default='configs/sweeps')
    ap.add_argument('--clean', action='store_true', help="remove existing configs first")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    configs = expand(spec)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.clean:
        for f in os.listdir(args.out_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(args.out_dir, f))

    by_axis = {}
    for cfg in configs:
        path = os.path.join(args.out_dir, f"{cfg['run_id']}.json")
        json.dump(cfg, open(path, 'w'), indent=2, sort_keys=True)
        by_axis.setdefault(cfg['axis'], []).append(cfg['run_id'])

    print(f"Wrote {len(configs)} configs to {args.out_dir}/\n")
    for axis in sorted(by_axis):
        print(f"  {axis:10s} {len(by_axis[axis]):3d}  e.g. {by_axis[axis][0]}")
    print(f"\nUnique encoders needing feature caches: "
          f"{sorted({c['fm'] for c in configs})}")


if __name__ == "__main__":
    main()
