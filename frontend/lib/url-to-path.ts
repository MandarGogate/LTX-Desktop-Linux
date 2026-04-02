import { getCachedAppDataDir } from './serve-url'
import { resolveRegisteredPath } from './url-path-registry'

/**
 * Extract a filesystem path from a media URL.
 * Supports:
 * - file:// URLs
 * - registered runtime URL↔path mappings
 * - /serve/files/<relative-path> URLs in web mode
 */
export function fileUrlToPath(url: string): string | null {
  const registeredPath = resolveRegisteredPath(url)
  if (registeredPath) return registeredPath

  if (url.startsWith('file://')) {
    let p = decodeURIComponent(url.slice(7)) // file:///Users/x -> /Users/x
    if (/^\/[A-Za-z]:/.test(p)) p = p.slice(1)
    return p
  }

  if (url.startsWith('/serve/files/')) {
    const appDataDir = getCachedAppDataDir()
    if (!appDataDir) return null
    const relative = decodeURIComponent(url.slice('/serve/files/'.length))
    return `${appDataDir.replace(/\/$/, '')}/${relative}`
  }

  return null
}
