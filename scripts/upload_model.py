import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

repo_id = "user/fire-moonshot-classifier"
model = Path("models/cnn_diversify_20260831_1429.inference.pt")
digest = hashlib.sha256(model.read_bytes()).hexdigest()

api = HfApi()
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True,
)

# 모델과 checksum을 같은 commit으로 업로드합니다.
commit = api.create_commit(
    repo_id=repo_id,
    repo_type="model",
    commit_message="Release Diversify inference model v1",
    operations=[
        CommitOperationAdd(
            path_in_repo=model.name,
            path_or_fileobj=model,
        ),
        CommitOperationAdd(
            path_in_repo=model.name + ".sha256",
            path_or_fileobj=f"{digest}  {model.name}\n".encode(),
        ),
    ],
)

# 배포할 모델 버전을 고정하는 설정 파일입니다.
manifest = {
    "repo_id": repo_id,
    "filename": model.name,
    "revision": commit.oid,
    "sha256": digest,
}
Path("fire-moonshot-classifier-model.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("Uploaded:", commit.commit_url)
print("Pinned revision:", commit.oid)
print("Saved: fire-moonshot-classifier-model.json")