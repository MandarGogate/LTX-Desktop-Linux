import { useCallback, useState } from 'react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'

export type IcLoraConditioningType = 'canny' | 'depth' | 'pose'

export interface IcLoraSubmitParams {
  videoPath: string
  conditioningType: IcLoraConditioningType
  conditioningStrength: number
  prompt: string
}

export interface IcLoraResult {
  videoPath: string
  videoUrl: string
}

interface UseIcLoraState {
  isGenerating: boolean
  status: string
  error: string | null
  result: IcLoraResult | null
}

export function useIcLora() {
  const [state, setState] = useState<UseIcLoraState>({
    isGenerating: false,
    status: '',
    error: null,
    result: null,
  })

  const submitIcLora = useCallback(async (params: IcLoraSubmitParams) => {
    if (!params.videoPath || !params.prompt.trim()) return

    setState({
      isGenerating: true,
      status: 'Generating',
      error: null,
      result: null,
    })

    try {
      const response = await backendFetch('/api/ic-lora/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: params.videoPath,
          conditioning_type: params.conditioningType,
          conditioning_strength: params.conditioningStrength,
          prompt: params.prompt,
        }),
      })

      const data = await response.json()
      if (response.ok && data.status === 'complete' && data.video_path) {
        const pathNormalized = data.video_path.replace(/\\/g, '/')
        const videoUrl = pathNormalized.startsWith('/') ? `file://${pathNormalized}` : `file:///${pathNormalized}`
        setState({
          isGenerating: false,
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
        status: '',
        error: errorMsg,
        result: null,
      })
    } catch (error) {
      const message = (error as Error).message || 'Unknown error'
      logger.error(`IC-LoRA error: ${message}`)
      setState({
        isGenerating: false,
        status: '',
        error: message,
        result: null,
      })
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      status: '',
      error: null,
      result: null,
    })
  }, [])

  return {
    submitIcLora,
    resetIcLora: reset,
    isIcLoraGenerating: state.isGenerating,
    icLoraStatus: state.status,
    icLoraError: state.error,
    icLoraResult: state.result,
  }
}
