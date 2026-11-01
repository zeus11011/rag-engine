import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = "http://localhost:8000";
// Proxy to FastAPI so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/documents": api, "/query": api, "/health": api } },
});
