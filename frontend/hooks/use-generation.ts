import { useState, useCallback, useRef } from 'react'
import type { GenerationSettings } from '../components/SettingsPanel'
import { backendFetch } from '../lib/backend'
import { toServableUrl } from '../lib/serve-url'

interface GenerationState {
  isGenerating: boolean
  progress: number
  statusMessage: string
  videoUrl: string | null
  videoPath: string | null  // Original file path for upscaling
  imageUrl: string | null
  imagePath: string | null  // Original file path for first image
  imageUrls: string[]  // For multiple image variations
  imagePaths: string[]  // Original file paths for all images
  error: string | null
}

interface GenerationProgress {
  status: string
  phase: string
  progress: number
  currentStep: number | null
  totalSteps: number | null
}

interface UseGenerationReturn extends GenerationState {
  generate: (prompt: string, imagePath: string | null, settings: GenerationSettings, audioPath?: string | null) => Promise<void>
  generateImage: (prompt: string, settings: GenerationSettings) => Promise<void>
  cancel: () => void
  reset: () => void
}

const IMAGE_SHORT_SIDE_BY_RESOLUTION: Record<string, number> = {
  '1080p': 1080,
  '1440p': 1440,
  '2048p': 2048,
}

const IMAGE_ASPECT_RATIO_VALUE: Record<string, number> = {
  '1:1': 1,
  '16:9': 16 / 9,
  '9:16': 9 / 16,
  '4:3': 4 / 3,
  '3:4': 3 / 4,
  '21:9': 21 / 9,
}

function getImageDimensions(settings: GenerationSettings): { width: number; height: number } {
  const shortSide = IMAGE_SHORT_SIDE_BY_RESOLUTION[settings.imageResolution]
  if (!shortSide) {
    throw new Error(`Unsupported image resolution mapping: ${settings.imageResolution}`)
  }

  const ratio = IMAGE_ASPECT_RATIO_VALUE[settings.imageAspectRatio]
  if (!ratio) {
    throw new Error(`Unsupported image aspect ratio mapping: ${settings.imageAspectRatio}`)
  }

  if (ratio >= 1) {
    return { width: Math.round(shortSide * ratio), height: shortSide }
  }
  return { width: shortSide, height: Math.round(shortSide / ratio) }
}

// Map phase to user-friendly message
function getPhaseMessage(phase: string): string {
  switch (phase) {
    case 'validating_request':
      return 'Validating request...'
    case 'uploading_image':
      return 'Uploading image...'
    case 'uploading_audio':
      return 'Uploading audio...'
    case 'loading_model':
      return 'Loading model...'
    case 'encoding_text':
      return 'Encoding prompt...'
    case 'inference':
      return 'Generating...'
    case 'downloading_output':
      return 'Downloading output...'
    case 'decoding':
      return 'Decoding video...'
    case 'complete':
      return 'Complete!'
    default:
      return 'Generating...'
  }
}

export function useGeneration(): UseGenerationReturn {
  const [state, setState] = useState<GenerationState>({
    isGenerating: false,
    progress: 0,
    statusMessage: '',
    videoUrl: null,
    videoPath: null,
    imageUrl: null,
    imagePath: null,
    imageUrls: [],
    imagePaths: [],
    error: null,
  })

  const abortControllerRef = useRef<AbortController | null>(null)

  const generate = useCallback(async (
    prompt: string,
    imagePath: string | null,
    settings: GenerationSettings,
    audioPath?: string | null,
  ) => {
    const statusMsg = settings.model === 'pro'
      ? 'Loading Pro model & generating...'
      : 'Generating video...'

    setState({
      isGenerating: true,
      progress: 0,
      statusMessage: statusMsg,
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imagePath: null,
      imageUrls: [],
      imagePaths: [],
      error: null,
    })

    abortControllerRef.current = new AbortController()
    let progressInterval: ReturnType<typeof setInterval> | null = null
    let shouldApplyPollingUpdates = true

    try {
      // Prepare JSON body
      const body: Record<string, unknown> = {
        prompt,
        model: settings.model,
        duration: String(settings.duration),
        resolution: settings.videoResolution,
        fps: String(settings.fps),
        audio: String(settings.audio),
        cameraMotion: settings.cameraMotion,
        aspectRatio: settings.aspectRatio || '16:9',
      }
      if (imagePath) {
        body.imagePath = imagePath
      }
      if (audioPath) {
        body.audioPath = audioPath
      }

      // Poll for real progress from backend
      
      const pollProgress = async () => {
        if (!shouldApplyPollingUpdates) return
        try {
          const res = await backendFetch('/api/generation/progress')
          if (res.ok) {
            const data: GenerationProgress = await res.json()
            if (!shouldApplyPollingUpdates) return

            let displayProgress = data.progress
            let statusMessage = getPhaseMessage(data.phase)
            
            // Use step-based progress during inference
            if (data.phase === 'inference' && data.totalSteps && data.totalSteps > 0) {
              const stepFraction = (data.currentStep ?? 0) / data.totalSteps
              // Map inference progress to 15%–90% range
              displayProgress = 15 + Math.floor(stepFraction * 75)
            } else if (data.phase === 'inference') {
              // No step info yet, show a gentle progress
              displayProgress = Math.max(displayProgress, 15)
            }

            // Backend sets progress=100 when complete; map to 95 here
            // so the final 100% only shows after HTTP response arrives
            if (data.phase === 'complete' || data.status === 'complete') {
              displayProgress = 95
              statusMessage = 'Finalizing...'
            }
            
            setState(prev => ({
              ...prev,
              progress: displayProgress,
              statusMessage,
            }))
          }
        } catch {
          // Ignore polling errors
        }
      }
      
      progressInterval = setInterval(pollProgress, 500)

      // Start generation (HTTP POST - synchronous, returns when done)
      const response = await backendFetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortControllerRef.current.signal,
      })
      shouldApplyPollingUpdates = false

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Generation failed')
      }

      const result = await response.json()
      
      if (result.status === 'complete' && result.video_path) {
        // Convert path to a URL the browser can load
        const videoPathNormalized = result.video_path.replace(/\\/g, '/')
        const fileUrl = videoPathNormalized.startsWith('/') ? `file://${videoPathNormalized}` : `file:///${videoPathNormalized}`
        
        setState({
          isGenerating: false,
          progress: 100,
          statusMessage: 'Complete!',
          videoUrl: toServableUrl(fileUrl),
          videoPath: result.video_path,
          imageUrl: null,
          imagePath: null,
          imageUrls: [],
          imagePaths: [],
          error: null,
        })
      } else if (result.status === 'cancelled') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else if (result.error) {
        throw new Error(result.error)
      }

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        }))
      }
    } finally {
      shouldApplyPollingUpdates = false
      if (progressInterval) {
        clearInterval(progressInterval)
      }
    }
  }, [])

  const cancel = useCallback(async () => {
    // Abort the fetch request
    abortControllerRef.current?.abort()
    
    // Also tell the backend to cancel
    try {
      await backendFetch('/api/generate/cancel', { method: 'POST' })
    } catch {
      // Ignore errors from cancel request
    }
    
    setState(prev => ({
      ...prev,
      isGenerating: false,
      statusMessage: 'Cancelled',
    }))
  }, [])

  const generateImage = useCallback(async (
    prompt: string,
    settings: GenerationSettings
  ) => {
    const numImages = settings.variations || 1
    
    setState({
      isGenerating: true,
      progress: 0,
      statusMessage: numImages > 1 ? `Generating ${numImages} images...` : 'Generating image...',
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imagePath: null,
      imageUrls: [],
      imagePaths: [],
      error: null,
    })

    abortControllerRef.current = new AbortController()

    try {
      // Skip prompt enhancement for T2I - use original prompt directly
      const finalPrompt = prompt

      const dims = getImageDimensions(settings)
      const numSteps = settings.imageSteps || 4

      // Poll for progress
      const pollProgress = async () => {
        try {
          const res = await backendFetch('/api/generation/progress')
          if (res.ok) {
            const data = await res.json()
            const currentStep = data.currentStep || 0
            const totalSteps = data.totalSteps || numSteps
            const stepText = data.phase === 'inference' && totalSteps > 0
              ? ` (${Math.min(currentStep, totalSteps)}/${totalSteps} steps)`
              : ''
            setState(prev => ({
              ...prev,
              progress: data.progress,
              statusMessage: data.phase === 'loading_model'
                ? 'Loading Z-Image Turbo model...'
                : data.phase === 'inference'
                  ? numImages > 1
                    ? `Generating images...${stepText}`
                    : `Generating image...${stepText}`
                  : data.phase === 'complete'
                    ? 'Complete!'
                    : 'Generating...',
            }))
          }
        } catch {
          // Ignore polling errors
        }
      }
      
      const progressInterval = setInterval(pollProgress, 500)

      let response: Response
      try {
        response = await backendFetch('/api/generate-image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: finalPrompt,
            width: dims.width,
            height: dims.height,
            numSteps,
            numImages,
          }),
          signal: abortControllerRef.current.signal,
        })
      } finally {
        clearInterval(progressInterval)
      }

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Image generation failed')
      }

      const result = await response.json()
      
      if (result.status === 'complete') {
        // Handle both new format (image_paths array) and old format (single image_path)
        let rawPaths: string[] = []
        if (result.image_paths && Array.isArray(result.image_paths)) {
          rawPaths = result.image_paths
        } else if (result.image_path) {
          rawPaths = [result.image_path]
        }
        
        if (rawPaths.length > 0) {
          // Convert all paths to servable URLs
          const fileUrls = rawPaths.map((path: string) => {
            const imagePath = path.replace(/\\/g, '/')
            const fileUrl = imagePath.startsWith('/') ? `file://${imagePath}` : `file:///${imagePath}`
            return toServableUrl(fileUrl)
          })
          
          setState({
            isGenerating: false,
            progress: 100,
            statusMessage: 'Complete!',
            videoUrl: null,
            videoPath: null,
            imageUrl: fileUrls[0],
            imagePath: rawPaths[0],
            imageUrls: fileUrls,    // All images
            imagePaths: rawPaths,   // All image paths
            error: null,
          })
        }
      } else if (result.status === 'cancelled') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else if (result.error) {
        throw new Error(result.error)
      }

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          statusMessage: 'Cancelled',
        }))
      } else {
        setState(prev => ({
          ...prev,
          isGenerating: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        }))
      }
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      isGenerating: false,
      progress: 0,
      statusMessage: '',
      videoUrl: null,
      videoPath: null,
      imageUrl: null,
      imagePath: null,
      imageUrls: [],
      imagePaths: [],
      error: null,
    })
  }, [])

  return {
    ...state,
    generate,
    generateImage,
    cancel,
    reset,
  }
}
