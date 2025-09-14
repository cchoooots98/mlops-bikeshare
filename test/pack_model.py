import os
import tarfile

os.makedirs(".model_dir", exist_ok=True)
src = r"mlruns\1\models\m-02fa82813dbe4fbcab848468b9d1e744\artifacts"

out = r"model_dir\model.tar.gz"

with tarfile.open(out, "w:gz") as tar:
    tar.add(src, arcname=".")

print("Packed:", os.path.abspath(out))
