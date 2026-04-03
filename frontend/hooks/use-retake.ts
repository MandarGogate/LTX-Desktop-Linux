import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'
import { toServableUrl } from '../lib/serve-url'

export type RetakeMode = 'replace_audio_and_video' | 'replace_video' | 'replace_audio'

export interface RetakeSubmitParams {
  videoPath: string
  startTime: number
  duration: number
  prompt: string
  mode: RetakeMode
  resolution: '540p' | '720p' | '1080p'
}

export interface RetakeResult {
  videoPath: string
  videoUrl: string
}

interface UseRetakeState {
  isRetaking: boolean
  progress: number
  retakeStatus: string
  retakeError: string | null
  result: RetakeResult | null
}

export function useRetake() {
  const [state, setState] = useState<UseRetakeState>({
    isRetaking: false,
    progress: 0,
    retakeStatus: '',
    retakeError: null,
    result: null,
  })

  const submitRetake = useCallback(async (params: RetakeSubmitParams) => {
    if (!params.videoPath) return

    setState({
      isRetaking: true,
      progress: 0,
      retakeStatus: 'Generating',
      retakeError: null,
      result: null,
    })

    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldPoll = true

    try {
      const pollProgress = async () => {
        if (!shouldPoll) return
        try {
          const res = await backendFetch('/api/generation/progress')
          if (!res.ok) return
          const data = await res.json() as { progress: number; phase: string; status: string }
          if (!shouldPoll) return
          let status = data.phase || 'Generating'
          let progress = data.progress || 0
          setState(prev => {
            if (data.phase === 'loading_model') status = 'Loading model...'
            else if (data.phase === 'inference') {
              status = 'Generating...'
              progress = Math.max(progress, Math.min(97, prev.progress + 2))
            } else if (data.phase === 'complete') {
              status = 'Finalizing...'
              progress = 95
            }
            return { ...prev, progress, retakeStatus: status }
          })
        } catch {
          // ignore polling errors
        }
      }
      progressInterval = setInterval(() => { void pollProgress() }, 500)

      const response = await backendFetch('/api/retake', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: params.videoPath,
          start_time: params.startTime,
          duration: params.duration,
          prompt: params.prompt,
          mode: params.mode,
          resolution: params.resolution,
        }),
      })

      shouldPoll = false
      const data = await response.json()

      if (response.ok && data.status === 'complete' && data.video_path) {
        const pathNormalized = data.video_path.replace(/\\/g, '/')
        const fileUrl = pathNormalized.startsWith('/') ? `file://${pathNormalized}` : `file:///${pathNormalized}`
        const videoUrl = toServableUrl(fileUrl)

        setState({
          isRetaking: false,
          progress: 100,
          retakeStatus: 'Retake complete!',
          retakeError: null,
          result: {
            videoPath: data.video_path,
            videoUrl,
          },
        })
        return
      }

      const errorMsg = data.error || 'Unknown error'
      setState({
        isRetaking: false,
        progress: 0,
        retakeStatus: '',
        retakeError: errorMsg,
        result: null,
      })
      logger.error(`Retake failed: ${errorMsg}`)
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`Retake error: ${message}`)
      setState({
        isRetaking: false,
        progress: 0,
        retakeStatus: '',
        retakeError: message,
        result: null,
      })
    } finally {
      shouldPoll = false
      if (progressInterval) clearInterval(progressInterval)
    }
  }, [])

  const resetRetake = useCallback(() => {
    setState({
      isRetaking: false,
      progress: 0,
      retakeStatus: '',
      retakeError: null,
      result: null,
    })
  }, [])

  return {
    submitRetake,
    resetRetake,
    isRetaking: state.isRetaking,
    retakeProgress: state.progress,
    retakeStatus: state.retakeStatus,
    retakeError: state.retakeError,
    retakeResult: state.result,
  }
}
