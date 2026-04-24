from huggingface_hub import HfApi

api = HfApi()

# Force Python 3.10
api.upload_file(
    path_or_fileobj="3.10\n".encode(),
    path_in_repo=".python-version",
    repo_id="amenuvevemelissa/ai-detector",
    repo_type="space"
)
print("Python version file uploaded!")

# Update requirements without gradio version pin
content = "gradio==4.44.0\nscikit-learn\npandas\nnumpy\njoblib\nxgboost\n"
api.upload_file(
    path_or_fileobj=content.encode(),
    path_in_repo="requirements.txt",
    repo_id="amenuvevemelissa/ai-detector",
    repo_type="space"
)
print("Requirements updated!")