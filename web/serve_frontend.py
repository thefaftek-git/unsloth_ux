


"""
Simple HTTP server to serve the frontend.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Change working directory to the frontend directory
        os.chdir(str(Path(__file__).parent / "frontend"))
        super().__init__(*args, **kwargs)

def run_server():
    """Run a simple HTTP server for serving the frontend."""
    print(f"Serving frontend on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("HTTP server running. Press Ctrl+C to stop.")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()


