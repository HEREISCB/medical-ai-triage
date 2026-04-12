module.exports = {
  apps: [
    {
      name: "triage-web",
      script: "uvicorn",
      args: "src.main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      cwd: "/home/ubuntu/medical-ai-triage",
      env_file: ".env",
      autorestart: true,
    },
    {
      name: "triage-agent",
      script: "python",
      args: "run.py start",
      interpreter: "none",
      cwd: "/home/ubuntu/medical-ai-triage",
      env_file: ".env",
      autorestart: true,
    },
  ],
};
