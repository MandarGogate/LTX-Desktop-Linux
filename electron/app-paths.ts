import { app } from 'electron'
import fs from 'fs'
import path from 'path'
import os from 'os'

export const APP_FOLDER_NAME = 'LTXDesktop'

function isDirectoryEmpty(dirPath: string): boolean {
  try {
    return fs.readdirSync(dirPath).length === 0
  } catch {
    return true
  }
}

function resolveLegacyUserDataPath(): string {
  return path.join(os.homedir(), '.ltx-desktop')
}

function resolveUserDataPath(): string {
  const envPath = process.env.LTX_APP_DATA_DIR?.trim()
  if (envPath) {
    return envPath
  }

  if (process.platform === 'win32') {
    const localAppData = process.env.LOCALAPPDATA
      || path.join(os.homedir(), 'AppData', 'Local')
    return path.join(localAppData, APP_FOLDER_NAME)
  }
  if (process.platform === 'darwin') {
    return path.join(
      os.homedir(),
      'Library',
      'Application Support',
      APP_FOLDER_NAME,
    )
  }
  const xdgData = process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share')
  const defaultPath = path.join(xdgData, APP_FOLDER_NAME)
  const legacyPath = resolveLegacyUserDataPath()

  // Reuse the legacy Linux data directory so existing model downloads keep working.
  if (fs.existsSync(legacyPath) && (!fs.existsSync(defaultPath) || isDirectoryEmpty(defaultPath))) {
    return legacyPath
  }

  return defaultPath
}

app.setPath('userData', resolveUserDataPath())

export function getAppDataDir(): string {
  return app.getPath('userData')
}

export function getLogDir(): string {
  return path.join(app.getPath('userData'), 'logs')
}
