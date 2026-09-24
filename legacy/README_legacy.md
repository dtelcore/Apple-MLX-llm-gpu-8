# Legacy English experiments (pre-0.1.0)

These runs used wiki prose (`data/train.txt`) and fact-inject / v10 mixes.
The wrong corpus does not teach open English. They are kept for inspection,
not as the active path.

`probe_mode=english` and `probe_mode=inject` stay in the unguided kernel so
old checkpoints can still be scored. Default `tests/` discovery does not
collect the files under `legacy/tests/`.

`data/train.txt` stays in the project `data/` directory and is **not** the
English corpus. Active English is TinyStories via
`python tools/prepare_tinystories.py` and
`setup/english_tinystories_c256_l6_config.json`.
