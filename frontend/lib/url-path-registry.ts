const urlToPathMap = new Map<string, string>()

export function registerUrlPath(url: string | null | undefined, path: string | null | undefined): void {
  if (!url || !path) return
  urlToPathMap.set(url, path)
}

export function resolveRegisteredPath(url: string | null | undefined): string | null {
  if (!url) return null
  return urlToPathMap.get(url) || null
}
