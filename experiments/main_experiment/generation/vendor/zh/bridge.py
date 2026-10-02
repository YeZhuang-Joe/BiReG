import hashlib
def sha(data): return hashlib.sha256(data).hexdigest()
