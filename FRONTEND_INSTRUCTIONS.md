# 📋 Production Deployment Instructions for Frontend Agent

**Objective:** Prepare this Next.js application for deployment in a secure "Hidden Backend" Docker architecture.

**Context:**
This frontend will run alongside a Python backend in a Docker Compose setup. The backend is **internal-only** and completely inaccessible from the public internet. All communication with the backend must happen safely server-side within the Docker network.

## Task Checklist

### 1. [ ] Configure Docker-Optimized Build
- Update `next.config.js` to enable standalone mode:
  ```javascript
  module.exports = {
    output: 'standalone',
    // ... other config
  }
  ```
- Create a production-grade `Dockerfile`:
  - Use a multi-stage build (deps -> builder -> runner) to minimize image size.
  - Expose port `3000`.
  - Ensure it runs as a non-root user (e.g., `nextjs` user) for security.

### 2. [ ] Configure Internal API Routing
- **IMPORTANT**: The browser cannot see the backend. Do not perform `fetch('http://backend:3001/...')` in `useEffect` or Client Components.
- Implement data fetching using **Server Actions** or **Route Handlers** (API Proxy).
- Use an environment variable for the backend URL (e.g., `INTERNAL_API_URL`).
- Default this variable to `http://backend:3001` (the internal Docker hostname).

### 3. [ ] Generate Docker Compose Snippet
- Provide a YAML service definition for `frontend` that I can paste into my main `docker-compose.prod.yml`.
- It should follow this pattern:
  ```yaml
  frontend:
    build: .
    restart: always
    expose:
      - "3000"
    environment:
      - INTERNAL_API_URL=http://backend:3001
  ```

---

**Definition of Done:**
- A valid `Dockerfile` exists in the root.
- `next.config.js` is updated for standalone output.
- All backend calls are confirmed to be server-side only.
- I have the `docker-compose` service snippet ready to copy.
