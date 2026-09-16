"""Verify the delivered experimental files without modifying them."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST_SHA256.json').read_text())
fail=[name for name,h in manifest.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=h]
if fail:raise SystemExit('Integrity failures: '+str(fail))
print(f'PASS: {len(manifest)} files match SHA-256 manifest')
