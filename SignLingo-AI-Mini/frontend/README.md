# SignLingo AI Mini — Frontend

React + Vite + Tailwind single-page UI for the ASL pose retrieval prototype.

## Commands

```bash
npm install
npm run dev      # http://localhost:5173 (proxies /api to http://localhost:8000)
npm run build    # type-check + production build
npm run lint
```

The backend must be running on port 8000 for uploads and retrieval to work.
Start it from the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

## Layout

- Left: educational video player
- Right: animated 2D ASL pose skeleton
- Bottom: transcript, upload controls, Generate / Compare
