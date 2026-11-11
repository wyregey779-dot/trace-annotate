"""A missing optional cv2 cannot prevent creation of a truthful model-run record."""
import json
from pathlib import Path
import subprocess
import sys


def test_face_code_identity_survives_missing_cv2_in_isolated_process():
    source = Path(__file__).resolve().parents[1] / 'trace_annotate/extractors/faces.py'
    program = r'''
import importlib.util,json,sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
# A fresh -I child has no inherited imports. Model an unavailable optional
# dependency without uninstalling or mutating the developer's environment.
sys.modules['cv2']=None
path=Path(sys.argv[1]);sys.path.insert(0,str(path.parents[2]))
spec=importlib.util.spec_from_file_location('face_optional_dependency_probe',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert module.HAS_CV2 is False
extractor=module.FaceExtractor()
identity=extractor.code_identity
assert identity['model_name']=='unavailable'
assert identity['model_version']=='unavailable'
assert identity['detector_kind'] is None
result=extractor.detect_faces(np.zeros((32,32,3),dtype=np.uint8))
assert result['status']=='not_run' and result['face_count'] is None
print(json.dumps({'isolated':bool(sys.flags.isolated),'identity':identity,'result':result}))
'''
    result = subprocess.run([sys.executable,'-I','-B','-c',program,str(source)],
                            capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode == 0,result.stdout+result.stderr
    assert json.loads(result.stdout)['isolated'] is True
