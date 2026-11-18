# scruple-provider-claude-cli

A Scruple `DecisionProvider` that answers each decision with `claude -p`, using your logged-in Claude Code session. You don't need an API key.

```ts
import { defineConfig } from '@scruple/core'
import { claudeCliProvider } from 'scruple-provider-claude-cli'

export default defineConfig({
  provider: claudeCliProvider(),
  // parser, plugins, rules
})
```

## Requirements

- `claude` is on `PATH` and logged in. Check with `claude -p hi`.
- Node 22.18 or later.

## Options

| Option | Default | Controls |
|---|---|---|
| `model` | `'claude-haiku-5-5'` | `--model`. A reply that any other model served is rejected. |
| `effort` | `'low'` | `--effort` |
| `concurrency` | `8` | How many `claude -p` processes Scruple runs at once |
| `authRetries` | `3` | Retries of a transient "Not logged in" failure, which concurrent `claude -p` processes report while the login is valid |
| `retryDelayMs` | `1000` | Delay before each retry, multiplied by the attempt number |

The provider id is `claude-cli:<model>:<effort>`. Scruple's decision cache uses it as part of each entry's key, so a change of model or effort skips the old cache entries.

## How a decision is asked

- `--system-prompt` tells Haiku to give a probability for every label, and to commit when the definitions settle the question.
- `-p` carries the request state and questions as JSON with sorted keys, so identical requests send identical prompts.
- `--json-schema` requires every question and every label.
- Each distribution is scaled to sum to 1. The top label becomes the answer, and its probability becomes the confidence.
- `--tools ''`, `--safe-mode`, and `--no-session-persistence` keep each call to a single answer with no tool use and no saved session.
