# Running the bot under systemd (Debian 12)

The unit file is `deepseek-trader.service`. It assumes the project lives at
`/opt/deepseek-trader` and runs as a dedicated `trader` user. Change the two
`/opt/deepseek-trader` paths and the `User=/Group=` lines if you use different
values.

## 1. Create a dedicated non-root user

Running a trading bot as root is unnecessary risk. Create a system user that
owns the code and has no login shell:

```bash
sudo adduser --system --group --home /opt/deepseek-trader --shell /usr/sbin/nologin trader
```

## 2. Put the code in place

```bash
# copy/clone the repo to /opt/deepseek-trader, then:
sudo chown -R trader:trader /opt/deepseek-trader
```

## 3. Build the virtualenv (as the trader user)

Debian 12 ships Python 3.11. Install the venv package if missing:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
sudo -u trader python3 -m venv /opt/deepseek-trader/venv
sudo -u trader /opt/deepseek-trader/venv/bin/pip install --upgrade pip
sudo -u trader /opt/deepseek-trader/venv/bin/pip install -r /opt/deepseek-trader/requirements.txt
```

## 4. Create `.env` with your live credentials

```bash
sudo -u trader cp /opt/deepseek-trader/.env.template /opt/deepseek-trader/.env
sudo -u trader nano /opt/deepseek-trader/.env      # fill in API keys etc.
sudo chmod 600 /opt/deepseek-trader/.env           # keys are secrets
```

The app reads `.env` itself (python-dotenv), so systemd does not need to.

## 5. (Optional) Redis

The unit soft-depends on `redis-server`. If your config uses Redis:

```bash
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

If you do not use Redis, ignore this — the `Wants=` is soft and the bot starts
without it.

## 6. Install and start the service

```bash
sudo cp /opt/deepseek-trader/deepseek-trader.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepseek-trader
```

## 7. Verify

```bash
systemctl status deepseek-trader          # is it active (running)?
journalctl -u deepseek-trader -f          # live logs (Ctrl-C to exit)
```

You should see the NautilusTrader banner and `AUTO_CONFIRM=true - Skipping user
confirmation`. The bot's own detailed JSON logs are still written under
`/opt/deepseek-trader/logs/`.

## Day-to-day

```bash
sudo systemctl stop deepseek-trader       # stop (sends SIGINT, 30s grace)
sudo systemctl restart deepseek-trader    # restart (e.g. after editing .env)
sudo systemctl disable deepseek-trader    # don't start on boot
journalctl -u deepseek-trader --since today
```

If a bad deploy makes it crash-loop, systemd stops retrying after 5 failures in
5 minutes. After fixing, clear the failed state:

```bash
sudo systemctl reset-failed deepseek-trader
sudo systemctl start deepseek-trader
```

## Notes / gotchas

- **`AUTO_CONFIRM=true` is mandatory** — without it `main_live.py` blocks on an
  interactive `input()` prompt that can never be answered under systemd.
- **Do not add `EnvironmentFile=.env`** — this `.env` has inline `# comments`
  that systemd would parse into the variable values, corrupting them. The app
  loads `.env` via python-dotenv instead.
- If hardening blocks something (e.g. you relocate logs outside the project),
  add the path to `ReadWritePaths=` or relax `ProtectSystem=`/`ProtectHome=`.
- This is a **live-money** service. Test with a paper/testnet key and
  `journalctl -f` before enabling on boot.
