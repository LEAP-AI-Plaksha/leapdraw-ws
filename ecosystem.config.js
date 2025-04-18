module.exports = {
  apps: [
    {
      name: "draw",
      script: "/home/niksh_hiremath24/leapdraw-ws/.venv/bin/uvicorn",
      args: "Backend:app --host 0.0.0.0 --port 443 " +
            "--ssl-keyfile /etc/letsencrypt/live/draw.leap-aiclub.tech/privkey.pem " +
            "--ssl-certfile /etc/letsencrypt/live/draw.leap-aiclub.tech/fullchain.pem",
      cwd: "/home/niksh_hiremath24/leapdraw-ws",
      interpreter: "none",
      watch: false,
    },
  ],
};
