#!/usr/bin/env bash
# Runs ON the OCI VM (Ubuntu). Idempotent: safe to re-run after every code
# sync -- it (re)installs deps, opens the port, (re)installs the systemd
# units and restarts the app. Normally invoked for you by deploy/oci/push.sh;
# can also be run by hand:  cd ~/TrendLine && bash deploy/oci/setup.sh
#
# Env overrides:  APP_DIR (default ~/TrendLine)   PORT (default 5000)
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/TrendLine}"
PORT="${PORT:-5000}"
RUN_USER="$(id -un)"
UNIT_SRC="$APP_DIR/deploy/oci"

echo "==> TrendLine setup: dir=$APP_DIR port=$PORT user=$RUN_USER"
cd "$APP_DIR"

# --- system packages (python3-venv is not on the stock OCI Ubuntu image) ---
if ! dpkg -s python3-venv >/dev/null 2>&1; then
    echo "==> installing python3-venv"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
fi

# --- virtualenv + deps ---
if [ ! -x venv/bin/python ]; then
    echo "==> creating venv"
    python3 -m venv venv
fi
echo "==> installing requirements"
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt

# --- runtime dirs / config ---
mkdir -p data cache
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "!!  created $APP_DIR/.env from .env.example -- fill in TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
fi

# --- OS firewall (OCI Ubuntu images ship an iptables REJECT-all after :22) ---
if ! sudo iptables -C INPUT -p tcp -m tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
    echo "==> opening tcp/$PORT in iptables"
    sudo iptables -I INPUT -p tcp -m tcp --dport "$PORT" -j ACCEPT
    sudo netfilter-persistent save >/dev/null
fi

# --- systemd units (fill placeholders, install, enable) ---
echo "==> installing systemd units"
for unit in trendline.service trendline-alerts.service trendline-alerts.timer; do
    sed -e "s|__APP_DIR__|$APP_DIR|g" \
        -e "s|__PORT__|$PORT|g" \
        -e "s|__USER__|$RUN_USER|g" \
        "$UNIT_SRC/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable trendline.service trendline-alerts.timer >/dev/null 2>&1
sudo systemctl restart trendline.service
sudo systemctl start trendline-alerts.timer

# --- report ---
sleep 3
echo
systemctl --no-pager --lines=0 status trendline.service | sed -n '1,4p'
systemctl --no-pager list-timers trendline-alerts.timer | sed -n '1,2p'
echo
echo "==> done. App: http://$(curl -s -m 5 ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}'):$PORT"
