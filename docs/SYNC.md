# Data Sync Guide

Push local Helix-DB data to the Fly.io production volume.

## Architecture

Ingestion and summarization run locally (CPU-heavy, hundreds of API calls). The Fly.io deployment only serves read-only MCP search queries. The `kidkazz db sync` command copies finished LMDB data files to the Fly volume.

```
Local Laptop                          Fly.io (production)
─────────────                         ──────────────────
1. kidkazz ingest markdown ...
2. kidkazz summarize generate ...
3. Local Helix-DB updated
       │
       ▼
4. kidkazz db sync                →   /data/user/ (Fly volume)
                                       Machine restarts
                                       MCP server serves new data
```

## Prerequisites

- **Fly CLI** installed: `curl -L https://fly.io/install.sh | sh`
- **Local data exists**: Run `kidkazz db deploy` and ingest at least one document
- **Fly app deployed**: `fly deploy --local-only` has been run at least once
- **Authenticated**: `fly auth login`

## Usage

### Preview (dry run)

```bash
kidkazz db sync --dry-run
```

Shows the sync plan without executing:
- Local data path and size
- Fly app name (from `fly.toml`)
- Steps that would be performed

### Sync

```bash
# Uses app name from fly.toml
kidkazz db sync

# Explicit app name
kidkazz db sync --app kidkazz-rag

# JSON output (for scripting)
kidkazz db sync --json
```

### What happens

1. Locates local LMDB data at `.helix/.volumes/dev/user/`
2. Verifies `fly` CLI is installed
3. Resolves Fly app name from `--app` flag or `fly.toml`
4. Stops the Fly machine (ensures clean LMDB state)
5. Creates a compressed tarball of the `user/` directory
6. Uploads via `fly ssh sftp` and extracts to `/data/user/`
7. Starts the machine and waits for health check

## When to Sync

- After ingesting new books (`kidkazz ingest markdown`)
- After generating or regenerating summaries (`kidkazz summarize generate`)
- After deduplicating data (`kidkazz db dedup-edges`, `kidkazz db dedup-summaries`)

## Manual Sync (alternative)

If the CLI command fails, you can sync manually:

```bash
# 1. Create tarball
cd .helix/.volumes/dev/
tar czf /tmp/helix-data.tar.gz user/

# 2. Stop machine
fly machine stop --app kidkazz-rag

# 3. Upload
fly ssh sftp shell --app kidkazz-rag
> put /tmp/helix-data.tar.gz /tmp/helix-data.tar.gz

# 4. Extract on remote
fly ssh console --app kidkazz-rag -C "tar xzf /tmp/helix-data.tar.gz -C /data/ && rm /tmp/helix-data.tar.gz"

# 5. Start machine
fly machine start --app kidkazz-rag
```

## Verifying the Sync

```bash
# Health check
curl https://kidkazz-rag.fly.dev/health

# Check document count via MCP
fly ssh console --app kidkazz-rag -C "curl -s http://localhost:6969/ListDocuments -d '[{}]' -H 'Content-Type: application/json' | head -c 200"
```

## Troubleshooting

### Machine won't stop
The machine may already be stopped (scale-to-zero). The sync command handles this gracefully.

### Upload timeout
Large databases (>500MB) may take a while. The default timeout is 5 minutes. For very large datasets, use the manual method with `scp` or split the tarball.

### Health check fails after sync
The machine uses scale-to-zero. After sync, the first request wakes it up (cold start ~15-30 seconds). Try:
```bash
curl https://kidkazz-rag.fly.dev/health
```

### Permission errors on remote
The Fly volume is mounted at `/data/`. Ensure the `user/` directory has correct permissions:
```bash
fly ssh console --app kidkazz-rag -C "ls -la /data/user/"
```

### Volume size
The default Fly volume is 1GB. Check usage:
```bash
fly ssh console --app kidkazz-rag -C "df -h /data"
```
To extend: `fly volumes extend <vol_id> --size 3` (size in GB).
