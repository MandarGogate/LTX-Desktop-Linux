export function pathToFileUrl(filePath: string): string {
  let normalized = filePath.replace(/\\/g, '/')
  if (!normalized.startsWith('/')) {
    normalized = `/${normalized}`
  }
  const encoded = normalized
    .split('/')
    .map(segment => encodeURIComponent(segment))
    .join('/')
  return `file://${encoded}`
}
