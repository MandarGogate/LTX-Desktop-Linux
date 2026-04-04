import { backendFetch } from './backend'

export class ApiClientError extends Error {
  status: number
  endpoint: string
  payload: unknown

  constructor(message: string, status: number, endpoint: string, payload: unknown) {
    super(message)
    this.name = 'ApiClientError'
    this.status = status
    this.endpoint = endpoint
    this.payload = payload
  }
}

function toErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== 'object') return fallback
  const record = payload as Record<string, unknown>
  if (typeof record.error === 'string' && record.error.trim()) return record.error
  if (typeof record.message === 'string' && record.message.trim()) return record.message
  if (typeof record.detail === 'string' && record.detail.trim()) return record.detail
  return fallback
}

async function requestJson(
  endpoint: string,
  method: 'GET' | 'POST',
  body?: unknown,
  init?: RequestInit,
): Promise<any> {
  const headers = new Headers(init?.headers)
  if (body !== undefined) {
    headers.set('Content-Type', 'application/json')
  }

  const response = await backendFetch(endpoint, {
    ...init,
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  })

  if (!response.ok) {
    let payload: unknown = null
    let fallback = `${response.status} ${response.statusText || 'Request failed'}`
    try {
      const text = await response.text()
      if (text) {
        fallback = text
        payload = JSON.parse(text) as unknown
        fallback = toErrorMessage(payload, fallback)
      }
    } catch {
      // Keep fallback.
    }
    throw new ApiClientError(fallback, response.status, endpoint, payload)
  }

  return response.json()
}

export class ApiClient {
  static suggestGapPrompt(body: unknown, init?: RequestInit): Promise<any> {
    return requestJson('/api/suggest-gap-prompt', 'POST', body, init)
  }

  static generateVideo(body: unknown, init?: RequestInit): Promise<any> {
    return requestJson('/api/generate', 'POST', body, init)
  }

  static generateImage(body: unknown, init?: RequestInit): Promise<any> {
    return requestJson('/api/generate-image', 'POST', body, init)
  }

  static retake(body: unknown): Promise<any> {
    return requestJson('/api/retake', 'POST', body)
  }

  static generateIcLora(body: unknown): Promise<any> {
    return requestJson('/api/ic-lora/generate', 'POST', body)
  }

  static extractIcLoraConditioning(body: unknown): Promise<any> {
    return requestJson('/api/ic-lora/extract-conditioning', 'POST', body)
  }
}
