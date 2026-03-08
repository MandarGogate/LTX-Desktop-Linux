import { logger } from './logger'

/**
 * Copy a generated file to the global project assets folder via a single IPC call.
 * Electron handles path validation, directory creation, and file copy.
 * Returns the new { path, url } if successful, or null on failure — callers handle fallback.
 */
export async function copyToAssetFolder(
  srcPath: string,
  projectId: string,
): Promise<{ path: string; url: string } | null> {
  if (!srcPath || !projectId || !window.electronAPI) return null
  try {
    const result = await window.electronAPI.copyToProjectAssets(srcPath, projectId)
    if (result.success && result.path && result.url) {
      return { path: result.path, url: result.url }
    }
    if (result.error) {
      logger.warn(`Failed to copy asset to project folder: ${result.error}`)
    }
  } catch (e) {
    logger.warn(`Failed to copy asset to project folder: ${e}`)
  }
  return null
}
