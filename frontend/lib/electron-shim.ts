/**
 * Electron API shim for web-mode operation.
 *
 * When the app runs outside Electron (as a standalone web app served by FastAPI),
 * this module provides a compatibility layer that implements every method on
 * `window.electronAPI` using HTTP calls to the backend instead of IPC.
 *
 * Import this before the app initializes — it will auto-install the shim
 * if `window.electronAPI` is not already defined (i.e., not running in Electron).
 */

const isElectron =
  typeof window !== 'undefined' &&
  typeof (window as any).electronAPI !== 'undefined'

/** Base URL for the backend (same origin in web mode) */
function getBaseUrl(): string {
  return window.location.origin
}

async function webFetch(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${getBaseUrl()}${path}`, init)
}

/**
 * Web-mode implementation of the Electron API.
 * Routes all calls through HTTP to the backend's /web/* endpoints.
 */
const webElectronAPI = {
  // ---- Backend connection ----
  getBackend: async () => ({
    url: getBaseUrl(),
    token: localStorage.getItem('ltx_auth_token') || '',
  }),

  // ---- File operations ----
  readLocalFile: async (filePath: string) => {
    const resp = await webFetch('/web/file/read', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: filePath }),
    })
    return resp.json()
  },

  showSaveDialog: async (_options: any): Promise<string | null> => {
    // In web mode, we can't open a native file dialog.
    // Return a default path and let the backend handle saving.
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    return `download-${timestamp}.mp4`
  },

  saveFile: async (
    filePath: string,
    data: string,
    encoding?: string,
  ): Promise<{ success: boolean; path?: string; error?: string }> => {
    const resp = await webFetch('/web/file/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: filePath, content: data, encoding: encoding || 'utf-8' }),
    })
    return resp.json()
  },

  // ---- App info ----
  getAppInfo: async () => {
    const resp = await webFetch('/web/app-info')
    const data = await resp.json()
    return {
      version: data.version,
      isPackaged: data.isPackaged,
      modelsPath: data.modelsPath,
      userDataPath: data.outputsPath,
    }
  },

  checkGpu: async () => {
    const resp = await webFetch('/web/gpu-info')
    return resp.json()
  },

  getModelsPath: async () => {
    const resp = await webFetch('/web/app-info')
    const data = await resp.json()
    return data.modelsPath
  },

  // ---- First-run (no-op in web mode) ----
  checkFirstRun: async () => ({ needsSetup: false, needsLicense: false }),
  acceptLicense: async () => true,
  completeSetup: async () => true,
  fetchLicenseText: async () => '',
  getNoticesText: async () => '',

  // ---- External links ----
  openLtxApiKeyPage: async () => {
    window.open('https://ltx.studio/api', '_blank')
    return true
  },
  openFalApiKeyPage: async () => {
    window.open('https://fal.ai/dashboard/keys', '_blank')
    return true
  },
  openParentFolderOfFile: async (_filePath: string) => {
    // Can't open file manager from browser
  },
  showItemInFolder: async (_filePath: string) => {
    // Can't open file manager from browser
  },

  // ---- Logs ----
  getLogs: async () => ({ logs: [] }),
  getLogPath: async () => ({ logPath: '', logDir: '' }),
  openLogFolder: async () => false,

  // ---- Resources ----
  getResourcePath: async () => null,
  getDownloadsPath: async () => '/tmp',

  // ---- Project assets ----
  copyToProjectAssets: async (
    _srcPath: string,
    _projectId: string,
  ) => ({
    success: false,
    error: 'Not supported in web mode — use upload instead',
  }),
  getProjectAssetsPath: async () => '',
  openProjectAssetsPathChangeDialog: async () => ({
    success: false,
    error: 'Not supported in web mode',
  }),

  // ---- Export (browser download) ----
  exportVideo: async (srcPath: string, _destPath: string) => {
    // Trigger a browser download
    const a = document.createElement('a')
    a.href = srcPath
    a.download = srcPath.split('/').pop() || 'video.mp4'
    a.click()
    return { success: true }
  },

  exportStill: async (srcPath: string, _destPath: string) => {
    const a = document.createElement('a')
    a.href = srcPath
    a.download = srcPath.split('/').pop() || 'image.png'
    a.click()
    return { success: true }
  },

  // ---- Updates (no-op in web mode) ----
  checkForUpdates: async () => ({
    available: false,
    version: '',
    releaseNotes: '',
  }),
  installUpdate: async () => false,
  onUpdateProgress: (_callback: any) => () => {},
  onUpdateStatus: (_callback: any) => () => {},

  // ---- Misc event listeners (no-op stubs) ----
  onSettingsChanged: (_callback: any) => () => {},
  onNavigate: (_callback: any) => () => {},
  onProjectChanged: (_callback: any) => () => {},
}

// ---- Auto-install shim if not in Electron ----
if (!isElectron) {
  ;(window as any).electronAPI = webElectronAPI
  console.log('[LTX Web Mode] Electron API shim installed')
}

export { isElectron, webElectronAPI }
