import { logger } from './logger'

export type ProjectAssetType = 'video' | 'image'

export interface ProjectAssetCopyResult {
  path: string
  bigThumbnailPath: string
  smallThumbnailPath: string
  width: number
  height: number
}

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

export async function addVisualAssetToProject(
  srcPath: string,
  projectId: string,
  type: ProjectAssetType,
): Promise<ProjectAssetCopyResult | null> {
  if (!srcPath || !projectId || !window.electronAPI?.copyToProjectAssets) return null
  try {
    const result = await window.electronAPI.copyToProjectAssets(srcPath, projectId)
    if (result.success && result.path) {
      void type
      return {
        path: result.path,
        bigThumbnailPath: '',
        smallThumbnailPath: '',
        width: 0,
        height: 0,
      }
    }
    if (result.error) {
      logger.warn(`Failed to add visual asset to project folder: ${result.error}`)
    }
  } catch (e) {
    logger.warn(`Failed to add visual asset to project folder: ${e}`)
  }
  return null
}

export async function addGenericAssetToProject(
  srcPath: string,
  projectId: string,
): Promise<{ path: string } | null> {
  if (!srcPath || !projectId || !window.electronAPI?.copyToProjectAssets) return null
  try {
    const result = await window.electronAPI.copyToProjectAssets(srcPath, projectId)
    if (result.success && result.path) {
      return { path: result.path }
    }
    if (result.error) {
      logger.warn(`Failed to copy file to project folder: ${result.error}`)
    }
  } catch (e) {
    logger.warn(`Failed to copy file to project folder: ${e}`)
  }
  return null
}
