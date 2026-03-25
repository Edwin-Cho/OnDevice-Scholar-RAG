# OnDevice Scholar RAG — Web UI

React 18 + Vite + Tailwind CSS frontend for the OnDevice Scholar RAG pipeline.

## Stack

- **React 18** + TypeScript
- **Vite** (dev server + build)
- **Tailwind CSS v4**
- **Lucide React** (icons)
- **Axios** (HTTP client)
- **React Router v6**

## Setup

```bash
npm install
npm run dev      # http://localhost:5173
```

Requires the FastAPI backend running at `http://localhost:8000`.

## Pages

| Route | Description |
|-------|-------------|
| `/login` | JWT authentication |
| `/` | Chat-style query interface with citation cards |
| `/documents` | PDF/TXT upload + delete |
| `/admin` | Index rebuild |

## Proxy

Vite proxies `/api/*` → `http://localhost:8000/*` in development.
