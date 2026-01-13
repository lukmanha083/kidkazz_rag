# Logging Guide

KidKazz RAG includes logging for debugging API calls and tracking system behavior.

## Enabling Logging

### Using the `--verbose` Flag

Add `-v` or `--verbose` to any CLI command to enable DEBUG level logging:

```bash
kidkazz -v inbox parse document.pdf -c block --agentic
kidkazz --verbose search semantic "query text"
```

### Using Environment Variable

Set `KIDKAZZ_LOG_LEVEL` to control log verbosity:

```bash
# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
export KIDKAZZ_LOG_LEVEL=DEBUG
kidkazz inbox parse document.pdf -c block

# Or inline for a single command
KIDKAZZ_LOG_LEVEL=INFO kidkazz inbox parse document.pdf
```

## Log Levels

| Level | What's Logged |
|-------|---------------|
| `DEBUG` | Full API response structure, block types, sample data |
| `INFO` | API call summaries (chunk count, block count, header count) |
| `WARNING` | Unexpected but non-fatal issues (default) |
| `ERROR` | API failures and exceptions |

## Reducto API Logging

When parsing PDFs, the following information is logged:

### INFO Level
```
Parsing PDF: document.pdf (chunk_mode=block, agentic=True)
Reducto API response: chunks=150, blocks=320, headers=25
```

### DEBUG Level
```
Uploading PDF to Reducto API...
Upload complete: file_id=abc123
Calling Reducto parse API...
Parse API call complete
Block types: {'Header': 25, 'Text': 280, 'Table': 15}
Sample block (dict): keys=['type', 'content', 'level']
```

## Debugging Header Extraction

If headers aren't being extracted, enable DEBUG logging to see the block structure:

```bash
KIDKAZZ_LOG_LEVEL=DEBUG kidkazz inbox parse document.pdf -c block 2>&1 | grep -E "(Block types|Sample block|headers=)"
```

Expected output when headers are detected:
```
Reducto API response: chunks=150, blocks=320, headers=25
Block types: {'Header': 25, 'Text': 280, 'Table': 15}
Sample block (dict): keys=['type', 'content', 'level']
```

If `headers=0`, check:
1. Block types - are there any `Header` blocks?
2. Sample block keys - does it have `type` and `content`?

## Log Output

Logs are written to `stderr` to avoid interfering with command output:

```bash
# Separate logs from output
kidkazz -v inbox parse doc.pdf 2>debug.log

# View logs while running
kidkazz -v inbox parse doc.pdf 2>&1 | tee debug.log
```

## MCP Server Logging

The MCP server also respects `KIDKAZZ_LOG_LEVEL`:

```bash
KIDKAZZ_LOG_LEVEL=DEBUG python -m src.mcp_server
```
