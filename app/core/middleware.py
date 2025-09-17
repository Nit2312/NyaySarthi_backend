import time
from collections import defaultdict
from fastapi.responses import JSONResponse

class RateLimitMiddleware:
    def __init__(self, app, max_requests=100, window=60):
        self.app = app
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
            
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        now = time.time()
        
        # Clean up old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < self.window]
        
        if len(self.requests[client_ip]) >= self.max_requests:
            response = JSONResponse(
                {"error": "Rate limit exceeded. Please try again later."},
                status_code=429
            )
            await response(scope, receive, send)
            return
            
        self.requests[client_ip].append(now)
        await self.app(scope, receive, send)
