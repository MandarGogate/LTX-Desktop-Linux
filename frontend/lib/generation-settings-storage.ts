import type { GenerationSettings } from '../components/SettingsPanel'

function isGenerationSettings(value: unknown): value is GenerationSettings {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.model === 'string' &&
    typeof record.duration === 'number' &&
    typeof record.videoResolution === 'string' &&
    typeof record.fps === 'number' &&
    typeof record.audio === 'boolean' &&
    typeof record.cameraMotion === 'string' &&
    typeof record.imageResolution === 'string' &&
    typeof record.imageAspectRatio === 'string' &&
    typeof record.imageSteps === 'number'
  )
}

export function loadGenerationSettings(key: string, defaults: GenerationSettings): GenerationSettings {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return { ...defaults }
    const parsed = JSON.parse(raw)
    if (!isGenerationSettings(parsed)) return { ...defaults }
    return { ...defaults, ...parsed }
  } catch {
    return { ...defaults }
  }
}

export function saveGenerationSettings(key: string, settings: GenerationSettings): void {
  try {
    localStorage.setItem(key, JSON.stringify(settings))
  } catch {
    // Ignore storage failures.
  }
}

export function loadStringSetting(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) || fallback
  } catch {
    return fallback
  }
}

export function saveStringSetting(key: string, value: string): void {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Ignore storage failures.
  }
}
