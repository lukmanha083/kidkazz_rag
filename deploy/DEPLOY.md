# Deploying KidKazz RAG to Fly.io

This guide walks you through deploying the MCP server + Helix-DB to Fly.io so you can use the KidKazz RAG tools from any machine via Claude Code's remote HTTP transport.

Fly.io's scale-to-zero feature means the app only runs when you're actively using it. When idle for ~5 minutes, the machine stops and you stop paying. When a new request arrives, Fly starts it back up automatically (~10-15s cold start).

**Estimated cost**: ~$2.45/month at 2 hours/day usage (shared-cpu-2x, 2GB RAM).

## Prerequisites

Before you begin, make sure you have:

1. **Fly.io CLI** installed and authenticated
2. **Docker Desktop** running (the build happens locally)
3. **Local Helix-DB image** built (from `kidkazz db deploy`)
4. **Data already ingested** into your local Helix-DB

### Install Fly CLI

```bash
# macOS / Linux
curl -L https://fly.io/install.sh | sh

# Verify
fly version
```

### Sign up / Log in

```bash
fly auth signup    # First time
fly auth login     # Returning user
```

### Verify Local Helix Image Exists

The Dockerfile copies the `helix-container` binary from your local Docker image. This image is built automatically when you run `kidkazz db deploy`.

```bash
docker images | grep helix-kidkazz_rag
```

You should see:

```
helix-kidkazz_rag-dev   dev   abc123   ...   ~500MB
```

If not, run `kidkazz db deploy` first to build the Helix image with compiled queries.

## Step 1: Create the Fly App

```bash
cd /path/to/kidkazz_rag

fly apps create kidkazz-rag
```

This registers the app name `kidkazz-rag` on Fly.io. The name must be globally unique — if taken, choose another name and update `app = "..."` in `fly.toml`.

## Step 2: Create a Persistent Volume

Helix-DB stores data in LMDB files under `/data`. A Fly volume persists this data across machine stop/start cycles.

```bash
fly volumes create kidkazz_data --region sin --size 1
```

- `--region sin`: Singapore (matches `primary_region` in `fly.toml`). Choose a region close to you — run `fly platform regions` to see options.
- `--size 1`: 1 GB. Two ingested textbooks use ~200MB. Increase if you plan to ingest more.

To change the region, also update `primary_region` in `fly.toml`.

## Step 3: Set Secrets (API Keys)

Fly secrets are encrypted environment variables. They're injected at runtime and never stored in your Docker image.

```bash
fly secrets set \
  CO_API_KEY=your_cohere_api_key \
  OPENAI_API_KEY=your_openai_api_key \
  KIDKAZZ_API_KEY=your_secret_mcp_auth_key
```

| Secret | Required | Purpose |
|--------|----------|---------|
| `CO_API_KEY` | Yes | Cohere API key for embeddings (embed-v4.0) and reranking (rerank-v3.5) |
| `OPENAI_API_KEY` | Optional | OpenAI key if using OpenAI embeddings or summarization LLM |
| `KIDKAZZ_API_KEY` | Yes | Bearer token for authenticating MCP HTTP requests |

**Generate a strong API key:**

```bash
# Option 1: openssl
openssl rand -hex 32

# Option 2: python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save this key — you'll need it for Claude Code's `.mcp.json` config in Step 6.

## Step 4: Build and Deploy

```bash
fly deploy --local-only
```

This command:
1. Builds the Docker image on your machine (not Fly's remote builders)
2. Copies the `helix-container` binary from the local Helix image
3. Installs Python dependencies
4. Pushes the image to Fly's registry
5. Creates a machine and starts it

The `--local-only` flag is required because the Dockerfile references a local Docker image (`helix-kidkazz_rag-dev:dev`) that doesn't exist on Fly's remote builders.

First deploy takes 3-5 minutes. Subsequent deploys are faster due to Docker layer caching.

### Verify Deployment

```bash
# Check machine status
fly status

# Check health endpoint
curl https://kidkazz-rag.fly.dev/health
# → {"status":"ok"}

# Check logs
fly logs
```

You should see logs like:

```
[entrypoint] Starting Helix-DB...
[entrypoint] Waiting for Helix-DB on port 6969...
[entrypoint] Helix-DB ready (3s)
[entrypoint] Starting MCP server...
Starting KidKazz RAG MCP server
Transport: streamable-http
HTTP endpoint: http://0.0.0.0:8080/mcp
Health check: http://0.0.0.0:8080/health
Auth: enabled
```

### Verify Auth Works

```bash
# Should return 401 (no auth)
curl -s -o /dev/null -w "%{http_code}" -X POST https://kidkazz-rag.fly.dev/mcp

# Should return 200 or MCP response (with auth)
curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Authorization: Bearer your_secret_mcp_auth_key" \
  https://kidkazz-rag.fly.dev/mcp

# Health is always public (no auth needed)
curl https://kidkazz-rag.fly.dev/health
```

## Step 5: Migrate Data

Your local Helix-DB has ingested data that needs to be copied to the Fly volume. The data lives in LMDB files under the `/data` directory.

### Option A: Copy via SFTP

```bash
# Open SFTP shell to the running machine
fly ssh sftp shell

# Upload the local Helix data directory
put .helix/.volumes/dev/user /data/user
```

### Option B: Copy via SSH + tar

```bash
# Tar up the local data
tar czf helix-data.tar.gz -C .helix/.volumes/dev user/

# Copy to machine
fly ssh sftp put helix-data.tar.gz /tmp/helix-data.tar.gz

# SSH in and extract
fly ssh console
tar xzf /tmp/helix-data.tar.gz -C /data/
rm /tmp/helix-data.tar.gz
exit
```

### Option C: Re-ingest from scratch

If you prefer a fresh start, SSH into the machine and run the ingestion pipeline directly. This requires the markdown files to be available on the machine.

After copying data, restart the machine to pick up the new files:

```bash
fly machine restart
```

## Step 6: Configure Claude Code

Add the remote MCP server to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "kidkazz-rag": {
      "type": "streamable-http",
      "url": "https://kidkazz-rag.fly.dev/mcp",
      "headers": {
        "Authorization": "Bearer your_secret_mcp_auth_key"
      }
    }
  }
}
```

Replace `your_secret_mcp_auth_key` with the same key you set as `KIDKAZZ_API_KEY` in Step 3.

Now start Claude Code in any project with this `.mcp.json` and the KidKazz RAG tools will be available remotely.

## How It Works

### Architecture

```
┌──────────────── Fly Machine ────────────────┐
│  tini (PID 1)                                │
│    └─ entrypoint.sh                          │
│        ├─ helix-container (port 6969, local) │
│        └─ MCP server (port 8080 → Fly Proxy) │
│                                              │
│  /data (persistent volume, LMDB files)       │
└──────────────────────────────────────────────┘
     ↑ Fly Proxy: HTTPS termination, auto-stop/auto-start
```

- **Single machine** runs both Helix-DB and the MCP server
- **entrypoint.sh** starts Helix-DB first, waits for it to be ready, then starts the MCP server
- **tini** handles PID 1 responsibilities (signal forwarding, zombie reaping)
- **Fly Proxy** terminates TLS, routes HTTPS traffic to port 8080

### Scale-to-Zero Lifecycle

1. **Idle**: No HTTP requests for ~5 minutes → Fly sends SIGTERM → tini forwards to both processes → machine stops
2. **Stopped**: Machine is not running, no cost (only volume storage: $0.15/GB/month)
3. **Wake**: HTTP request arrives → Fly starts the machine → entrypoint boots Helix (~3-5s) → MCP server starts (~1s) → request is served
4. **Active**: Machine stays running while requests keep arriving

Cold start is ~10-15 seconds (Helix-DB startup dominates).

### Thread Limiting

Helix-DB defaults to spawning `nproc * 8` threads. On a 2-vCPU Fly machine, that's 16 threads — reasonable. But we set explicit limits via environment variables to be safe:

- `HELIX_CORES_OVERRIDE=2` — Gateway threads: 2*8 = 16
- `TOKIO_WORKER_THREADS=2` — Tokio async runtime
- `RAYON_NUM_THREADS=2` — Rayon parallel compute

### Authentication

The `ApiKeyMiddleware` checks `Authorization: Bearer <token>` on all HTTP requests except `/health` (which must be public for Fly's health checks). It uses constant-time comparison (`hmac.compare_digest`) to prevent timing attacks.

## Common Operations

### View Logs

```bash
fly logs              # Stream live logs
fly logs --no-tail    # Show recent logs and exit
```

### SSH into Machine

```bash
fly ssh console
```

### Restart Machine

```bash
fly machine restart
```

### Scale VM Size

Edit `fly.toml`:

```toml
[[vm]]
  size = "shared-cpu-4x"    # Upgrade from 2x
  memory = "4gb"             # More RAM
```

Then redeploy:

```bash
fly deploy --local-only
```

### Change Region

```bash
# Destroy existing volume (data will be lost!)
fly volumes list
fly volumes destroy vol_xxxxx

# Create volume in new region
fly volumes create kidkazz_data --region nrt --size 1

# Update fly.toml
# primary_region = "nrt"

# Redeploy
fly deploy --local-only
```

### Update API Keys

```bash
fly secrets set CO_API_KEY=new_key
# Machine restarts automatically after setting secrets
```

### Rotate MCP Auth Key

```bash
# Generate new key
openssl rand -hex 32

# Set on Fly
fly secrets set KIDKAZZ_API_KEY=new_key_here

# Update .mcp.json in all projects using this server
```

### Redeploy After Code Changes

After modifying the MCP server code or Helix queries:

```bash
# If Helix queries changed, rebuild the local image first
kidkazz db deploy

# Then deploy to Fly
fly deploy --local-only
```

### Destroy Everything

```bash
fly apps destroy kidkazz-rag
```

This removes the app, machines, and volumes. Data is permanently deleted.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `fly deploy` fails with image not found | Run `kidkazz db deploy` to build the local Helix image first |
| Health check failing | Check logs with `fly logs` — Helix-DB may be slow to start. Increase `grace_period` in `fly.toml` |
| 401 on MCP requests | Verify `KIDKAZZ_API_KEY` matches your `.mcp.json` `Authorization` header |
| Cold start too slow | Increase VM size or set `min_machines_running = 1` in `fly.toml` (costs more) |
| Volume full | `fly volumes extend vol_xxxxx --size 2` to grow to 2 GB |
| Data lost after deploy | Data lives on the volume at `/data`, not the image. Deploys don't touch volumes. If you recreated the volume, you need to re-migrate data |
| Connection refused on port 6969 | Helix-DB only listens on localhost inside the machine. The MCP server connects to it internally — you can't reach it from outside |
| `KIDKAZZ_API_KEY` not set warning | You deployed without setting secrets. Run `fly secrets set KIDKAZZ_API_KEY=...` |

## Cost Breakdown

| Resource | Cost | Notes |
|----------|------|-------|
| shared-cpu-2x (2 vCPU, 2GB RAM) | ~$0.0000694/sec | Only when running |
| 1 GB persistent volume | $0.15/month | Always (even when stopped) |
| Outbound bandwidth | $0.02/GB after 100GB free | Negligible for MCP traffic |

**Example**: 2 hours/day active usage = 2 * 3600 * $0.0000694 * 30 = **~$15/month** for compute + $0.15 volume = **~$15.15/month**.

With scale-to-zero and typical sporadic MCP usage (a few minutes at a time), actual cost is much lower.
