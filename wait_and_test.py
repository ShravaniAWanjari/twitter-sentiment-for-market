import requests
import time
import subprocess
import sys

def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(url)
            if res.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

if __name__ == "__main__":
    server_url = "http://127.0.0.1:8000/api/config"
    print(f"Waiting for server at {server_url}...")
    if wait_for_server(server_url):
        print("Server is UP! Running tests...")
        # Run verify_api.py
        result = subprocess.run([sys.executable, "verify_api.py"], capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    else:
        print("Server failed to start within timeout.")
