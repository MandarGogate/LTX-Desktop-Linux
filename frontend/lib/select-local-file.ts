import { persistSelectableFile } from './web-file-upload'

interface SelectLocalFileOptions {
  title: string
  extensions: string[]
  accept: string
  kind: 'image' | 'audio' | 'video' | 'generic'
}

function isWebMode(): boolean {
  return !(window as any).__ELECTRON__
}

function pathToFileUrl(filePath: string): string {
  const normalized = filePath.replace(/\\/g, '/')
  return normalized.startsWith('/') ? `file://${normalized}` : `file:///${normalized}`
}

async function selectViaBrowserPicker(options: SelectLocalFileOptions): Promise<{ path: string; url: string } | null> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = options.accept
    input.style.display = 'none'

    input.onchange = async () => {
      try {
        const file = input.files?.[0]
        if (!file) {
          resolve(null)
          return
        }
        const persisted = await persistSelectableFile(file, options.kind)
        resolve(persisted)
      } catch (error) {
        reject(error)
      } finally {
        input.remove()
      }
    }

    input.oncancel = () => {
      input.remove()
      resolve(null)
    }

    document.body.appendChild(input)
    input.click()
  })
}

export async function selectLocalFile(options: SelectLocalFileOptions): Promise<{ path: string; url: string } | null> {
  if (!isWebMode() && window.electronAPI?.showOpenFileDialog) {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: options.title,
      filters: [{ name: 'Files', extensions: options.extensions }],
    })

    if (paths && paths.length > 0) {
      const filePath = paths[0]
      return { path: filePath, url: pathToFileUrl(filePath) }
    }
  }

  return selectViaBrowserPicker(options)
}

export async function persistDroppedFile(
  file: File,
  kind: 'image' | 'audio' | 'video' | 'generic',
): Promise<{ path: string; url: string }> {
  return persistSelectableFile(file, kind)
}
