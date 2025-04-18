import hashlib
import threading

# Global cache and lock for thread-safe access
cache_model_hash = {}
_cache_lock = threading.Lock()


def calc_hash(filename):
    # Thread-safe check for existing hash
    with _cache_lock:
        if filename in cache_model_hash:
            return cache_model_hash[filename]

    try:
        sha256_hash = hashlib.sha256()

        # Efficient chunked reading
        with open(filename, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        model_hash = sha256_hash.hexdigest()[:10]

        # Store result in cache (thread-safe)
        with _cache_lock:
            cache_model_hash[filename] = model_hash

        return model_hash

    except Exception:
        return ""  # Return empty string on read failure
