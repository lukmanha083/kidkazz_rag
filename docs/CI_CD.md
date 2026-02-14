# CI/CD Pipeline

Automated testing via GitHub Actions. Deployment stays manual from the local machine.

## Architecture

```
GitHub Actions (CI)                   Local Laptop (CD)
───────────────────                   ─────────────────
Push / PR to main                     fly deploy --local-only
  → Install deps                        (Dockerfile depends on
  → Run pytest                           locally-built helix image)
  → Report results
                                      kidkazz db sync
                                        (push LMDB data to Fly volume)
```

**Why local deploy?** The Dockerfile at `deploy/Dockerfile` depends on a locally-built Helix-DB image. Building Helix from source requires Rust compilation that takes too long in CI. Once a container registry is set up, this can move to CI.

## GitHub Actions Workflow

File: `.github/workflows/test.yml`

### Triggers
- **Push** to `main` branch
- **Pull requests** targeting `main`

### What it runs
1. Checks out the code
2. Sets up Python 3.13
3. Installs all dependencies: `pip install -e ".[all,dev]"`
4. Downloads NLTK data (needed for extractive summarization tests)
5. Runs: `PYTHONPATH=. pytest tests/ -v -m "not helix and not mcp"`

### Skipped tests
- `@pytest.mark.helix` — Requires a running Helix-DB server
- `@pytest.mark.mcp` — MCP server tests (need Helix)

These tests run locally before deploying.

### Secrets

| Secret | Required | Purpose |
|--------|----------|---------|
| `OPENAI_API_KEY` | Yes | Embedding tests use real OpenAI API |

Add via: GitHub repo → Settings → Secrets → Actions → New repository secret

## Development Workflow

### Day-to-day development

```bash
# 1. Make changes locally
# 2. Run tests
PYTHONPATH=. pytest tests/ -v

# 3. Push — CI runs automatically
git push origin main
```

### Deploying code changes

```bash
# Build and deploy (requires local helix image)
fly deploy --local-only
```

### Syncing data

```bash
# After ingesting new books or regenerating summaries
kidkazz db sync
```

### Full release (code + data)

```bash
# 1. Run full test suite locally (including helix tests)
PYTHONPATH=. pytest tests/ -v

# 2. Deploy code
fly deploy --local-only

# 3. Push data
kidkazz db sync
```

## Future: Full CI/CD

When a container registry is set up, the workflow could be extended:

```yaml
# Not yet implemented — requires pushing helix image to registry
deploy:
  needs: test
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  steps:
    - uses: superfly/flyctl-actions/setup-flyctl@master
    - run: flyctl deploy --remote-only
      env:
        FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

This would require:
1. Push the Helix-DB base image to a registry (Docker Hub, GitHub Container Registry)
2. Update `deploy/Dockerfile` to pull from registry instead of local image
3. Add `FLY_API_TOKEN` secret to GitHub
4. Data sync would still be manual (local → Fly volume)
