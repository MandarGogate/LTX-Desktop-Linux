import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'
import { toServableUrl } from '../lib/serve-url'

export type IcLoraConditioningType = 'canny' | 'depth' | 'pose'
export type IcLoraModelType = 'union'

export interface IcLoraSubmitParams {
  videoPath: string
  imagePath?: string | null
  modelType: IcLoraModelType
  conditioningType: IcLoraConditioningType
  conditioningStrength: number
  resolution: '540p' | '720p' | '1080p'
  aspectRatio: '16:9' | '9:16'
  duration?: number | null
  prompt: string
}

export interface IcLoraResult {
  videoPath: string
  videoUrl: string
}

interface UseIcLoraState {
  isGenerating: boolean
  progress: number
  status: string
  error: string | null
  result: IcLoraResult | null
}

export function useIcLora() {
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    progress: 0,
    status: '',
    error: null,
    result: null,
  })

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams) => {
    if (!params.videoPath || !params.prompt.trim()) return

    setState({
      isGenerating: true,
      progress: 0,
      status: 'Generating',
      error: null,
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
          const data = await res.json() as { progress: number; phase: string; currentStep?: number | null; totalSteps?: number | null; status: string }
          if (!shouldPoll) return
          let progress = data.progress || 0
          let status = data.phase || 'Generating'
          setState(prev => {
            if (data.phase === 'preprocessing' && data.totalSteps && data.currentStep !== undefined && data.currentStep !== null) {
              status = `Preprocessing control video (${data.currentStep}/${data.totalSteps})`
            } else if (data.phase === 'loading_model') {
              status = 'Loading model...'
            } else if (data.phase === 'denoising_stage_1') {
              status = data.totalSteps && data.currentStep !== undefined && data.currentStep !== null
                ? `Denoising stage 1 (${data.currentStep}/${data.totalSteps})`
                : 'Denoising stage 1...'
              progress = Math.max(prev.progress, progress)
            } else if (data.phase === 'denoising_stage_2') {
              status = data.totalSteps && data.currentStep !== undefined && data.currentStep !== null
                ? `Denoising stage 2 (${data.currentStep}/${data.totalSteps})`
                : 'Denoising stage 2...'
              progress = Math.max(prev.progress, progress)
            } else if (data.phase === 'inference') {
              status = 'Generating...'
              progress = Math.max(prev.progress, progress)
            } else if (data.phase === 'retrying_low_vram') {
              status = 'Retrying in low-VRAM mode...'
              progress = Math.max(prev.progress, Math.min(70, progress))
            } else if (data.phase === 'complete') {
              status = 'Finalizing...'
              progress = 95
            }
            return { ...prev, progress, status }
          })
        } catch {
          // ignore polling errors
        }
      }

      progressInterval = setInterval(() => { void pollProgress() }, 500)

      const response = await backendFetch('/api/ic-lora/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: params.videoPath,
          model_type: params.modelType,
          conditioning_type: params.conditioningType,
          conditioning_strength: params.conditioningStrength,
          resolution: params.resolution,
          aspect_ratio: params.aspectRatio,
          duration: params.duration ?? null,
          prompt: params.prompt,
          images: params.imagePath ? [{ path: params.imagePath, frame: 0, strength: 1.0 }] : [],
        }),
      })

      shouldPoll = false
      const data = await response.json()
      if (response.ok && data.status === 'complete' && data.video_path) {
        const pathNormalized = data.video_path.replace(/\\/g, '/')
        const fileUrl = pathNormalized.startsWith('/') ? `file://${pathNormalized}` : `file:///${pathNormalized}`
        const videoUrl = toServableUrl(fileUrl)
        setState({
          isGenerating: false,
          progress: 100,
          status: 'Generation complete!',
          error: null,
          result: {
            videoPath: data.video_path,
            videoUrl,
          },
        })
        return
      }

      const errorMsg = data.error || 'Unknown error'
      logger.error(`IC-LoRA failed: ${errorMsg}`)
      setState({
        isGenerating: false,
        progress: 0,
        status: '',
        error: errorMsg,
        result: null,
      })
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`IC-LoRA error: ${message}`)
      setState({
        isGenerating: false,
        progress: 0,
        status: '',
        error: message,
        result: null,
      })
    } finally {
      shouldPoll = false
      if (progressInterval) clearInterval(progressInterval)
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      progress: 0,
      status: '',
      error: null,
      result: null,
    })
  }, [])

  return {
    submitIcLora,
    resetIcLora: reset,
    isIcLoraGenerating: state.isGenerating,
    icLoraProgress: state.progress,
    icLoraStatus: state.status,
    icLoraError: state.error,
    icLoraResult: state.result,
  }
}
