import requests
import os
import logging

class DownloaderTool:
    def download_file(self, url: str, filename: str) -> str:
        if '..' in filename or filename.startswith('/'):
            return "Error: Invalid filename. Path traversal is not allowed."
        sandbox_path = os.path.abspath("sandbox")
        full_path = os.path.join(sandbox_path, filename)
        if not os.path.abspath(full_path).startswith(sandbox_path):
            return "Error: Access denied. Attempted to save file outside the sandbox."
        logging.info(f"Attempting to download from {url} to {full_path}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            with requests.get(url, stream=True, timeout=30, headers=headers) as r:
                r.raise_for_status()
                with open(full_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return f"Successfully downloaded file from {url} and saved as '{filename}'."
        except Exception as e:
            return f"Error downloading file: {e}"