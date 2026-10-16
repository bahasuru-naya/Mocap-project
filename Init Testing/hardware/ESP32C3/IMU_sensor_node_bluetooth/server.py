import http.server
import socketserver
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

class ReuseTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

print("=" * 60)
print("  HTTP Server for IMU Visualization")
print("=" * 60)
print(f"Serving from: {DIRECTORY}")
print(f"Open browser: http://127.0.0.1:{PORT}")
print("=" * 60)
print("Server is running... Press Ctrl+C to stop")
print("=" * 60)

try:
    with ReuseTCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\n✓ Server stopped")
    sys.exit(0)
except OSError as e:
    print(f"\n✗ Error: {e}")
    print("Port 8000 might already be in use. Try:")
    print("  netstat -ano | findstr :8000  (Windows)")
    print("  lsof -i :8000  (Mac/Linux)")
    sys.exit(1)