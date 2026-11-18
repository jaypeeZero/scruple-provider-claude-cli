import test from 'ava'
import type { DecisionRequest } from '@scruple/core'
import { claudeCliProvider, describeCliFailure, type Exec } from '../src/index.ts'

const request: DecisionRequest = {
  state: { function: { text: 'let counter = 0', truncated: false }, zeta: 1, alpha: [true, null] },
  questions: {
    verdict: {
      type: 'choice',
      instructions: 'Does it read module state?',
      criteria: { hidden_state: 'Reads module scope', explicit_inputs: 'Only parameters' },
    },
    severity: {
      type: 'score',
      instructions: 'How bad?',
      criteria: ['harmless', 'smelly', 'broken'],
    },
    leaks: {
      type: 'noul',
      instructions: 'Does it leak a handle?',
      criteria: { true: 'A handle stays open', false: 'Every handle closes' },
    },
  },
}

const fullAnswer = {
  verdict: { hidden_state: 0.9, explicit_inputs: 0.1 },
  severity: { '0': 0.2, '1': 0.5, '2': 0.3 },
  leaks: { true: 0.25, false: 0.75 },
}

const cliOutput = (overrides: Record<string, unknown> = {}): string =>
  JSON.stringify({
    is_error: false,
    subtype: 'success',
    structured_output: fullAnswer,
    usage: { input_tokens: 2, cache_creation_input_tokens: 1550, cache_read_input_tokens: 40, output_tokens: 169 },
    modelUsage: { 'claude-haiku-5-5': {} },
    ...overrides,
  })

interface Call {
  args: string[]
  signal: AbortSignal | undefined
}

const recordingExec = (calls: Call[], output: () => string = () => cliOutput()): Exec =>
  async (args, signal) => {
    calls.push({ args, signal })
    return output()
  }

const flagValue = (args: string[], flag: string): string | undefined => args[args.indexOf(flag) + 1]

test('asks claude -p for Haiku at low effort with no tools and no saved session', async (t) => {
  const calls: Call[] = []

  await claudeCliProvider({ exec: recordingExec(calls) }).evaluate(request)

  const args = calls[0]!.args
  t.is(flagValue(args, '--model'), 'claude-haiku-5-5')
  t.is(flagValue(args, '--effort'), 'low')
  t.is(flagValue(args, '--output-format'), 'json')
  t.is(flagValue(args, '--tools'), '')
  t.true(args.includes('--safe-mode'))
  t.true(args.includes('--no-session-persistence'))
})

test('sends every question, label, criterion, and the state in the prompt', async (t) => {
  const calls: Call[] = []

  await claudeCliProvider({ exec: recordingExec(calls) }).evaluate(request)

  const prompt = flagValue(calls[0]!.args, '-p') ?? ''
  for (const text of ['Does it read module state?', 'Reads module scope', 'Only parameters', 'harmless', 'broken', 'A handle stays open', 'Every handle closes', 'let counter = 0']) {
    t.true(prompt.includes(text), `prompt is missing ${text}`)
  }
})

test('sends a schema that requires every question and every label of each', async (t) => {
  const calls: Call[] = []

  await claudeCliProvider({ exec: recordingExec(calls) }).evaluate(request)

  const schema = JSON.parse(flagValue(calls[0]!.args, '--json-schema') ?? '{}')
  t.deepEqual(schema.required, ['verdict', 'severity', 'leaks'])
  t.deepEqual(schema.properties.verdict.required, ['hidden_state', 'explicit_inputs'])
  t.deepEqual(schema.properties.severity.required, ['0', '1', '2'])
  t.deepEqual(schema.properties.leaks.required, ['true', 'false'])
  t.false(schema.properties.verdict.additionalProperties)
})

test('the system prompt tells the model to commit when the definitions settle the question', async (t) => {
  const calls: Call[] = []

  await claudeCliProvider({ exec: recordingExec(calls) }).evaluate(request)

  const system = flagValue(calls[0]!.args, '--system-prompt') ?? ''
  t.true(system.includes('insufficient_context'))
  t.true(system.includes('0.9 or more'))
})

test('identical requests with keys in another order send identical prompts', async (t) => {
  const calls: Call[] = []
  const provider = claudeCliProvider({ exec: recordingExec(calls) })

  await provider.evaluate(request)
  await provider.evaluate({ ...request, state: { alpha: [true, null], zeta: 1, function: { truncated: false, text: 'let counter = 0' } } })

  t.is(flagValue(calls[0]!.args, '-p'), flagValue(calls[1]!.args, '-p'))
})

test('answers choice, score, and noul questions from the distributions, with the model and summed input tokens', async (t) => {
  const response = await claudeCliProvider({ exec: recordingExec([]) }).evaluate(request)

  t.deepEqual(response, {
    model: 'claude-haiku-5-5',
    usage: { inputTokens: 1592, outputTokens: 169 },
    answers: {
      verdict: { type: 'choice', choice: 'hidden_state', confidence: 0.9, probabilities: { hidden_state: 0.9, explicit_inputs: 0.1 } },
      severity: { type: 'score', score: 1, confidence: 0.5, probabilities: { '0': 0.2, '1': 0.5, '2': 0.3 }, legend: { '0': 'harmless', '1': 'smelly', '2': 'broken' } },
      leaks: { type: 'noul', noul: 0.25 },
    },
  })
})

test('a distribution that does not sum to one is scaled to sum to one', async (t) => {
  const exec = recordingExec([], () => cliOutput({ structured_output: { ...fullAnswer, verdict: { hidden_state: 3, explicit_inputs: 1 } } }))

  const response = await claudeCliProvider({ exec }).evaluate(request)

  t.deepEqual(response.answers['verdict'], { type: 'choice', choice: 'hidden_state', confidence: 0.75, probabilities: { hidden_state: 0.75, explicit_inputs: 0.25 } })
})

test('an all-zero distribution is an error, not an invented answer', async (t) => {
  const exec = recordingExec([], () => cliOutput({ structured_output: { ...fullAnswer, verdict: { hidden_state: 0, explicit_inputs: 0 } } }))

  await t.throwsAsync(claudeCliProvider({ exec }).evaluate(request), { message: /verdict.*no probability mass/ })
})

test('a missing question or label is an error naming it', async (t) => {
  const missingQuestion = recordingExec([], () => cliOutput({ structured_output: { verdict: fullAnswer.verdict, severity: fullAnswer.severity } }))
  const missingLabel = recordingExec([], () => cliOutput({ structured_output: { ...fullAnswer, verdict: { hidden_state: 1 } } }))

  await t.throwsAsync(claudeCliProvider({ exec: missingQuestion }).evaluate(request), { message: /leaks/ })
  await t.throwsAsync(claudeCliProvider({ exec: missingLabel }).evaluate(request), { message: /verdict\.explicit_inputs/ })
})

test('an answer served by any model other than the requested one is an error naming that model', async (t) => {
  const fellBack = recordingExec([], () => cliOutput({ modelUsage: { 'claude-haiku-5-5': {}, 'claude-opus-5-5': {} } }))
  const wrongModel = recordingExec([], () => cliOutput({ modelUsage: { 'claude-opus-5-5': {} } }))
  const noUsage = recordingExec([], () => cliOutput({ modelUsage: undefined }))

  await t.throwsAsync(claudeCliProvider({ exec: fellBack }).evaluate(request), { message: /claude-opus-5-5/ })
  await t.throwsAsync(claudeCliProvider({ exec: wrongModel }).evaluate(request), { message: /claude-opus-5-5/ })
  await t.throwsAsync(claudeCliProvider({ exec: noUsage }).evaluate(request), { message: /modelUsage/ })
})

test('a CLI-reported error or a reply with no structured output is an error', async (t) => {
  const errored = recordingExec([], () => cliOutput({ is_error: true, result: 'boom' }))
  const empty = recordingExec([], () => cliOutput({ structured_output: undefined }))

  await t.throwsAsync(claudeCliProvider({ exec: errored }).evaluate(request), { message: /boom/ })
  await t.throwsAsync(claudeCliProvider({ exec: empty }).evaluate(request), { message: /structured_output/ })
})

const notLoggedIn = (): Error =>
  new Error('claude -p exit 1; stdout tail: {"is_error":true,"result":"Not logged in · Please run /login"}')

test('a transient not-logged-in failure is retried and the later answer returned', async (t) => {
  const attempts: number[] = []
  const exec: Exec = async () => {
    attempts.push(attempts.length)
    if (attempts.length < 3) throw notLoggedIn()
    return cliOutput()
  }

  const response = await claudeCliProvider({ exec, retryDelayMs: 0 }).evaluate(request)

  t.is(attempts.length, 3)
  t.is(response.answers['verdict']?.type, 'choice')
})

test('a login failure that persists past the retry budget is an error', async (t) => {
  const attempts: number[] = []
  const exec: Exec = async () => {
    attempts.push(attempts.length)
    throw notLoggedIn()
  }

  await t.throwsAsync(claudeCliProvider({ exec, retryDelayMs: 0, authRetries: 2 }).evaluate(request), { message: /Not logged in/ })
  t.is(attempts.length, 3)
})

test('a failure other than the login failure is not retried', async (t) => {
  const attempts: number[] = []
  const exec: Exec = async () => {
    attempts.push(attempts.length)
    throw new Error('claude -p exit 1; stdout tail: {"result":"API Error: 400 bad request"}')
  }

  await t.throwsAsync(claudeCliProvider({ exec, retryDelayMs: 0 }).evaluate(request), { message: /400/ })
  t.is(attempts.length, 1)
})

test('the abort signal reaches the CLI call', async (t) => {
  const calls: Call[] = []
  const controller = new AbortController()

  await claudeCliProvider({ exec: recordingExec(calls) }).evaluate(request, controller.signal)

  t.is(calls[0]!.signal, controller.signal)
})

test('the configured model, effort, and concurrency reach the call and the provider id', async (t) => {
  const calls: Call[] = []
  const exec = recordingExec(calls, () => cliOutput({ modelUsage: { 'claude-sonnet-5-5': {} } }))

  const provider = claudeCliProvider({ exec, model: 'claude-sonnet-5-5', effort: 'medium', concurrency: 4 })
  await provider.evaluate(request)

  t.is(provider.id, 'claude-cli:claude-sonnet-5-5:medium')
  t.is(provider.concurrency, 4)
  t.is(flagValue(calls[0]!.args, '--model'), 'claude-sonnet-5-5')
  t.is(flagValue(calls[0]!.args, '--effort'), 'medium')
})

test('a CLI failure message carries the exit code and the tail of stdout, not the prompt', (t) => {
  const failure = Object.assign(new Error('Command failed: claude -p SECRET-PROMPT-TEXT ...'), {
    code: 1,
    stdout: `${'x'.repeat(2000)}{"is_error":true,"result":"API Error: 529 overloaded"}`,
  })

  const message = describeCliFailure(failure)

  t.regex(message, /exit 1/)
  t.regex(message, /529 overloaded/)
  t.false(message.includes('SECRET-PROMPT-TEXT'))
  t.true(message.length < 1200)
})
