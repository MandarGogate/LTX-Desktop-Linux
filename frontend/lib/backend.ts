let cached: { url: string; token: string } | null = null

/**
 * Detect if running in web mode (no Electron) vs Electron mode.
 * In web mode, the backend is at the same origin.
 */
const isWebMode = typeof window !== 'undefined' && !(window as any).__ELECTRON__

export async function getBackendCredentials(): Promise<{ url: string; token: string }> {
  if (!cached) {
    if (isWebMode && typeof window !== 'undefined' && window.location) {
      // Web mode: backend is same origin
      cached = {
        url: window.location.origin,
        token: localStorage.getItem('ltx_auth_token') || '',
      }
    } else {
      cached = await window.electronAPI.getBackend()
    }
  }
  return cached
}

export function resetBackendCredentials(): void {
  cached = null
}

export async function backendFetch(path: string, init?: RequestInit): Promise<Response> {
  const { url, token } = await getBackendCredentials()
  const headers = new Headers(init?.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${url}${path}`, { ...init, headers })
}

export async function backendWsUrl(path: string): Promise<string> {
  const { url, token } = await getBackendCredentials()
  const ws = url.replace('http://', 'ws://')
  const sep = path.includes('?') ? '&' : '?'
  return `${ws}${path}${sep}token=${token}`
}
