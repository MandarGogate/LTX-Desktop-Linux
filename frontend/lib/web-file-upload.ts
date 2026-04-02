import { backendFetch } from './backend'
import { toServableUrl } from './serve-url'
import { registerUrlPath } from './url-path-registry'

export async function persistSelectableFile(
  file: File,
  kind: 'image' | 'audio' | 'video' | 'generic' = 'generic',
): Promise<{ url: string; path: string }> {
  const electronFilePath = (file as any).path as string | undefined
  // Only use Electron file path if it's actually an absolute path (starts with / or drive letter)
  if (electronFilePath && (electronFilePath.startsWith('/') || /^[A-Za-z]:/.test(electronFilePath))) {
    const normalized = electronFilePath.replace(/\\/g, '/')
    const url = normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
    registerUrlPath(url, electronFilePath)
    return { url, path: electronFilePath }
  }

  const formData = new FormData()
  formData.append('file', file)
  formData.append('kind', kind)

  const response = await backendFetch('/web/file/upload', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(await response.text() || 'Failed to upload file')
  }

  const payload = await response.json() as { path: string; url: string }
  const url = toServableUrl(payload.url)
  registerUrlPath(url, payload.path)
  return { url, path: payload.path }
}
