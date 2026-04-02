/**
 * Convert a local file path to a serveable HTTP URL in web mode.
 *
 * In Electron mode, `file://` URLs work natively. In web mode (browser),
 * the browser blocks `file://` for security, so we rewrite paths to go
 * through the backend's static file server at `/serve/files/` or
 * `/serve/outputs/`.
 */

const _isWebMode =
  typeof window !== 'undefined' && !(window as any).__ELECTRON__

/** The app data directory, e.g. `/home/user/.ltx-desktop` */
let _appDataDir: string | null = null

export function getCachedAppDataDir(): string | null {
  return _appDataDir
}

async function ensureAppDataDir(): Promise<string> {
  if (_appDataDir) return _appDataDir
  try {
    const info = await window.electronAPI.getAppInfo()
    // outputsPath is typically `~/.ltx-desktop/outputs`, parent is the app data dir
    const outputsPath: string = (info as any).modelsPath || ''
    // modelsPath = /home/user/.ltx-desktop/models → parent = /home/user/.ltx-desktop
    _appDataDir = outputsPath.replace(/\/models\/?$/, '')
    if (!_appDataDir) _appDataDir = '/home'
  } catch {
    _appDataDir = '/home'
  }
  return _appDataDir
}

/**
 * Convert a file path or file:// URL to a URL the browser can load.
 *
 * - In Electron mode: returns the original file:// URL (works natively).
 * - In web mode: rewrites to `/serve/files/<relative-path>` which is
 *   proxied to the backend's StaticFiles mount.
 */
export function toServableUrl(pathOrUrl: string): string {
  if (!_isWebMode) return pathOrUrl

  // Strip file:// prefix
  let filePath = pathOrUrl
  if (filePath.startsWith('file://')) {
    filePath = decodeURIComponent(filePath.slice(7))
    // Windows: file:///C:/... → C:/...
    if (/^\/[A-Za-z]:/.test(filePath)) filePath = filePath.slice(1)
  }

  // If it's already a browser-loadable URL, leave it alone
  if (
    filePath.startsWith('http://') ||
    filePath.startsWith('https://') ||
    filePath.startsWith('data:') ||
    filePath.startsWith('blob:') ||
    filePath.startsWith('/serve/')
  ) {
    return filePath
  }

  // Rewrite to the backend static file server.
  // The backend mounts /serve/files → ~/.ltx-desktop (the app data dir).
  // So /home/user/.ltx-desktop/outputs/video.mp4
  //  → /serve/files/outputs/video.mp4
  if (_appDataDir && filePath.startsWith(_appDataDir)) {
    const relative = filePath.slice(_appDataDir.length).replace(/^\//, '')
    return `/serve/files/${relative}`
  }

  // Fallback: try outputs
  const outputsMatch = filePath.match(/outputs\/(.+)$/)
  if (outputsMatch) {
    return `/serve/outputs/${outputsMatch[1]}`
  }

  // Last resort: serve the full path through the file read endpoint
  return `/serve/files/${filePath.replace(/^\//, '')}`
}

/**
 * Initialize the URL rewriter (call once at app startup in web mode).
 */
export async function initServableUrls(): Promise<void> {
  if (_isWebMode) {
    await ensureAppDataDir()
  }
}
