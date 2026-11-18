import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type {
  DecisionAnswer,
  DecisionProvider,
  DecisionQuestion,
  DecisionRequest,
  DecisionResponse,
  JsonValue,
} from '@scruple/core'

export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export type Exec = (args: string[], signal: AbortSignal | undefined) => Promise<string>

export interface ClaudeCliProviderOptions {
  model?: string
  effort?: Effort
  concurrency?: number
  authRetries?: number
  retryDelayMs?: number
  exec?: Exec
}

interface CliResult {
  is_error: boolean
  result?: string
  structured_output?: unknown
  usage: {
    input_tokens: number
    cache_creation_input_tokens?: number
    cache_read_input_tokens?: number
    output_tokens: number
  }
  modelUsage?: Record<string, unknown>
}

const SYSTEM_PROMPT = [
  'You are a code-review decision model.',
  'The user message is a JSON object with `state` (the only evidence you may use) and `questions`.',
  'Each label\'s description is its full definition. Match the evidence to those definitions; judge the code as shown.',
  'For each question, give a probability for every label: how likely that label is the correct answer given the evidence.',
  'The probabilities for one question sum to 1.',
  'Abstention labels such as `insufficient_context` are for code that lacks facts the definitions require. If the shown code and the definitions settle the question, put your probability on the label they point to.',
  'Calibration: when the evidence clearly matches one definition, put 0.9 or more on that label. Use 0.6 to 0.8 when the evidence fits two definitions. Spread probability further only when the definitions overlap on this code.',
  'Score questions use an ordered scale: label 0 is the first criterion.',
].join('\n')

const execFileAsync = promisify(execFile)
const MAX_STDOUT_BYTES = 8 * 1024 * 1024
const STDOUT_TAIL_CHARS = 800

/** Summarizes a failed `claude -p` run by exit code and stdout tail; the command line holds the prompt, so it is dropped. */
export const describeCliFailure = (error: unknown): string => {
  const failure = error as { code?: unknown, stdout?: unknown }
  const stdout = typeof failure.stdout === 'string' ? failure.stdout.trim() : ''
  const tail = stdout.length > STDOUT_TAIL_CHARS ? stdout.slice(-STDOUT_TAIL_CHARS) : stdout
  return `claude -p exit ${String(failure.code ?? 'unknown')}; stdout tail: ${tail.length === 0 ? '(empty)' : tail}`
}

const runClaude: Exec = async (args, signal) => {
  const pending = execFileAsync('claude', args, {
    maxBuffer: MAX_STDOUT_BYTES,
    ...(signal === undefined ? {} : { signal }),
  })
  // Without EOF on stdin, `claude -p` waits 3 s for piped input on every call.
  pending.child.stdin?.end()
  try {
    const { stdout } = await pending
    return stdout
  } catch (error) {
    if (signal?.aborted === true) throw error
    throw new Error(describeCliFailure(error), { cause: error })
  }
}

// Concurrent `claude -p` processes intermittently report "Not logged in" within
// ~100 ms while the stored login is valid; a short wait and retry succeeds.
const isTransientLoginFailure = (error: unknown): boolean =>
  error instanceof Error && error.message.includes('Not logged in')

const delay = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

const withLoginRetry = async (
  run: () => Promise<string>,
  retriesLeft: number,
  retryDelayMs: number,
  signal: AbortSignal | undefined,
  attempt = 1,
): Promise<string> => {
  try {
    return await run()
  } catch (error) {
    if (retriesLeft === 0 || signal?.aborted === true || !isTransientLoginFailure(error)) throw error
    await delay(retryDelayMs * attempt)
    return withLoginRetry(run, retriesLeft - 1, retryDelayMs, signal, attempt + 1)
  }
}

const labelsFor = (question: DecisionQuestion): Record<string, JsonValue> => {
  if (question.type === 'choice') {
    return question.criteria
  }
  if (question.type === 'score') {
    return Object.fromEntries(question.criteria.map((criterion, index) => [String(index), criterion]))
  }
  return {
    true: question.criteria?.true ?? 'The statement in the instructions is true.',
    false: question.criteria?.false ?? 'The statement in the instructions is false.',
  }
}

const stableStringify = (value: JsonValue): string => {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableStringify(entry)).join(',')}]`
  }
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key] ?? null)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

const renderRequest = (request: DecisionRequest): string =>
  stableStringify({
    state: request.state,
    questions: Object.fromEntries(
      Object.entries(request.questions).map(([name, question]) => [
        name,
        {
          type: question.type === 'noul' ? 'boolean' : question.type,
          instructions: question.instructions,
          labels: labelsFor(question),
        },
      ]),
    ),
  })

const answerSchema = (questions: Record<string, DecisionQuestion>): Record<string, unknown> => ({
  type: 'object',
  additionalProperties: false,
  required: Object.keys(questions),
  properties: Object.fromEntries(
    Object.entries(questions).map(([name, question]) => {
      const labels = Object.keys(labelsFor(question))
      return [
        name,
        {
          type: 'object',
          additionalProperties: false,
          required: labels,
          properties: Object.fromEntries(labels.map((label) => [label, { type: 'number' }])),
        },
      ]
    }),
  ),
})

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const distribution = (name: string, value: unknown, labels: readonly string[]): Record<string, number> => {
  if (!isRecord(value)) {
    throw new TypeError(`decision output is missing question ${name}`)
  }
  const raw = labels.map((label) => {
    const probability = value[label]
    if (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0) {
      throw new TypeError(`decision output ${name}.${label} must be a non-negative number`)
    }
    return [label, probability] as const
  })
  const total = raw.reduce((sum, [, probability]) => sum + probability, 0)
  if (total === 0) {
    throw new TypeError(`decision output ${name} has no probability mass`)
  }
  return Object.fromEntries(raw.map(([label, probability]) => [label, probability / total]))
}

const toAnswer = (question: DecisionQuestion, probabilities: Record<string, number>): DecisionAnswer => {
  if (question.type === 'noul') {
    return { type: 'noul', noul: probabilities['true'] ?? 0 }
  }
  const [top, confidence] = Object.entries(probabilities).reduce((best, entry) =>
    entry[1] > best[1] ? entry : best,
  )
  if (question.type === 'choice') {
    return { type: 'choice', choice: top, confidence, probabilities }
  }
  return { type: 'score', score: Number(top), confidence, probabilities, legend: labelsFor(question) }
}

const structuredOutputOf = (result: CliResult, model: string): Record<string, unknown> => {
  if (result.is_error) {
    throw new Error(`claude -p failed: ${result.result ?? 'unknown error'}`)
  }
  if (result.structured_output === undefined) {
    throw new Error('claude -p returned no structured_output')
  }
  if (result.modelUsage === undefined) {
    throw new Error('claude -p returned no modelUsage; cannot confirm which model answered')
  }
  const servedBy = Object.keys(result.modelUsage)
  if (servedBy.some((served) => served !== model) || !servedBy.includes(model)) {
    throw new Error(`claude -p was asked for ${model} but modelUsage lists ${servedBy.join(', ')}`)
  }
  if (!isRecord(result.structured_output)) {
    throw new TypeError('decision output must be a JSON object')
  }
  return result.structured_output
}

/** A Scruple decision provider that answers through `claude -p`, using the logged-in Claude Code session. */
export const claudeCliProvider = (options: ClaudeCliProviderOptions = {}): DecisionProvider => {
  const model = options.model ?? 'claude-haiku-5-5'
  const effort = options.effort ?? 'low'
  const exec = options.exec ?? runClaude
  const authRetries = options.authRetries ?? 3
  const retryDelayMs = options.retryDelayMs ?? 1000

  return {
    id: `claude-cli:${model}:${effort}`,
    concurrency: options.concurrency ?? 8,

    async evaluate (request: DecisionRequest, signal?: AbortSignal): Promise<DecisionResponse> {
      const args = [
        '-p', renderRequest(request),
        '--model', model,
        '--effort', effort,
        '--system-prompt', SYSTEM_PROMPT,
        '--json-schema', JSON.stringify(answerSchema(request.questions)),
        '--output-format', 'json',
        '--tools', '',
        '--safe-mode',
        '--no-session-persistence',
      ]
      const stdout = await withLoginRetry(() => exec(args, signal), authRetries, retryDelayMs, signal)
      const result = JSON.parse(stdout) as CliResult
      const output = structuredOutputOf(result, model)
      const answers = Object.fromEntries(
        Object.entries(request.questions).map(([name, question]) => [
          name,
          toAnswer(question, distribution(name, output[name], Object.keys(labelsFor(question)))),
        ]),
      )
      const { usage } = result
      return {
        model,
        answers,
        usage: {
          inputTokens: usage.input_tokens + (usage.cache_creation_input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0),
          outputTokens: usage.output_tokens,
        },
      }
    },
  }
}
