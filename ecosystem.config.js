module.exports = {
  apps: [
    {
      name: "draw",
      script: "/home/niksh_hiremath24/leapdraw-ws/.venv/bin/uvicorn",
      args: "Backend:app --host 0.0.0.0 --port 8080 " +
            "--ssl-keyfile /home/niksh_hiremath24/leapdraw-ws/certs/privkey.pem " +
            "--ssl-certfile /home/niksh_hiremath24/leapdraw-ws/certs/fullchain.pem",
      cwd: "/home/niksh_hiremath24/leapdraw-ws",
      interpreter: "none",
      watch: false,
    },
  ],
};
