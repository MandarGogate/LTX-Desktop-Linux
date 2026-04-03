import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Upload, Loader2, Film, Sparkles,
  RefreshCw, Download, AlertCircle, Trash2,
} from 'lucide-react'
import { backendFetch } from '../lib/backend'
import { logger } from '../lib/logger'
import { fileUrlToPath } from '../lib/url-to-path'
import { persistDroppedFile, selectLocalFile } from '../lib/select-local-file'
import { Select } from './ui/select'

export type ICLoraModelType = 'union' | 'motion_track'
export type ICLoraConditioningType = 'canny' | 'depth' | 'pose' | 'motion_track'

type DownloadStatus = 'idle' | 'downloading' | 'complete' | 'error'

interface IcLoraDownloadProgress {
  status: DownloadStatus
  current_downloading_file: string | null
  current_file_progress: number
  total_progress: number
  completed_files: string[]
  all_files: string[]
  error: string | null
}

interface ModelDownloadStartResponse {
  status?: string
  error?: string
  message?: string
  sessionId?: string
}

interface ModelsStatusModel {
  id: string
  downloaded: boolean
}

interface ModelsStatusResponse {
  models: ModelsStatusModel[]
}

interface ICLoraPanelProps {
  initialVideoUrl?: string | null
  initialVideoPath?: string | null
  initialImageUrl?: string | null
  initialImagePath?: string | null
  resetKey?: number
  fillHeight?: boolean
  isProcessing?: boolean
  processingStatus?: string
  modelType?: ICLoraModelType
  onModelTypeChange?: (type: ICLoraModelType) => void
  conditioningType?: ICLoraConditioningType
  onConditioningTypeChange?: (type: ICLoraConditioningType) => void
  conditioningStrength?: number
  onConditioningStrengthChange?: (strength: number) => void
  resolution?: '540p' | '720p' | '1080p'
  onResolutionChange?: (resolution: '540p' | '720p' | '1080p') => void
  aspectRatio?: '16:9' | '9:16'
  onAspectRatioChange?: (aspectRatio: '16:9' | '9:16') => void
  duration?: number | null
  onDurationChange?: (duration: number | null) => void
  outputVideoUrl?: string | null
  outputVideoPath?: string | null
  showInlineOutputPreview?: boolean
  onChange?: (data: {
    videoUrl: string | null
    videoPath: string | null
    imageUrl: string | null
    imagePath: string | null
    modelType: ICLoraModelType
    conditioningType: ICLoraConditioningType
    conditioningStrength: number
    resolution: '540p' | '720p' | '1080p'
    aspectRatio: '16:9' | '9:16'
    duration: number | null
    ready: boolean
  }) => void
}

export const IC_LORA_MODEL_TYPES: { value: ICLoraModelType; label: string; desc: string }[] = [
  { value: 'union', label: 'Union Control', desc: 'Canny, depth, or pose guidance' },
  { value: 'motion_track', label: 'Motion Track', desc: 'Video with colored spline / trajectory overlays' },
]

export const CONDITIONING_TYPES: { value: ICLoraConditioningType; label: string; desc: string }[] = [
  { value: 'canny', label: 'Canny Edges', desc: 'Edge detection' },
  { value: 'depth', label: 'Depth Map', desc: 'Estimated depth' },
  { value: 'pose', label: 'Pose', desc: 'OpenPose-style skeleton guidance' },
  { value: 'motion_track', label: 'Motion Track', desc: 'Trajectory overlay control video' },
]

export function getConditioningTypesForModel(modelType: ICLoraModelType) {
  return modelType === 'motion_track'
    ? CONDITIONING_TYPES.filter(ct => ct.value === 'motion_track')
    : CONDITIONING_TYPES.filter(ct => ct.value !== 'motion_track')
}

const IC_LORA_MODEL_IDS = ['ic_lora', 'ic_lora_motion_track', 'depth_processor', 'person_detector', 'pose_processor'] as const
type IcLoraModelId = typeof IC_LORA_MODEL_IDS[number]

const IC_LORA_MODEL_LABELS: Record<IcLoraModelId, string> = {
  ic_lora: 'IC-LoRA Union Control',
  ic_lora_motion_track: 'IC-LoRA Motion Track',
  depth_processor: 'Depth Processor',
  person_detector: 'Person Detector',
  pose_processor: 'Pose Processor',
}

const EMPTY_IC_MODEL_STATUS: Record<IcLoraModelId, boolean> = {
  ic_lora: false,
  ic_lora_motion_track: false,
  depth_processor: false,
  person_detector: false,
  pose_processor: false,
}

export function ICLoraPanel({
  initialVideoUrl,
  initialVideoPath,
  initialImageUrl,
  initialImagePath,
  resetKey,
  fillHeight = false,
  isProcessing = false,
  processingStatus = '',
  modelType: modelTypeProp,
  onModelTypeChange,
  conditioningType: conditioningTypeProp,
  onConditioningTypeChange,
  conditioningStrength: conditioningStrengthProp,
  onConditioningStrengthChange,
  resolution = '540p',
  onResolutionChange,
  aspectRatio = '16:9',
  onAspectRatioChange,
  duration = null,
  onDurationChange,
  outputVideoUrl,
  outputVideoPath: _outputVideoPath,
  showInlineOutputPreview = true,
  onChange,
}: ICLoraPanelProps) {
  const inputVideoRef = useRef<HTMLVideoElement>(null)
  const [inputVideoUrl, setInputVideoUrl] = useState<string | null>(initialVideoUrl || null)
  const [inputVideoPath, setInputVideoPath] = useState<string | null>(initialVideoPath || null)
  const [inputImageUrl, setInputImageUrl] = useState<string | null>(initialImageUrl || null)
  const [inputImagePath, setInputImagePath] = useState<string | null>(initialImagePath || null)
  const [inputTime, setInputTime] = useState(0)

  const [internalModelType, setInternalModelType] = useState<ICLoraModelType>('union')
  const [internalCondType, setInternalCondType] = useState<ICLoraConditioningType>('canny')
  const [internalCondStrength, setInternalCondStrength] = useState(1.0)
  const modelType = modelTypeProp ?? internalModelType
  const conditioningType = conditioningTypeProp ?? internalCondType
  const conditioningStrength = conditioningStrengthProp ?? internalCondStrength
  const [conditioningPreview, setConditioningPreview] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)

  const [icModelDownloaded, setIcModelDownloaded] = useState<Record<IcLoraModelId, boolean>>({ ...EMPTY_IC_MODEL_STATUS })
  const [isCheckingIcLora, setIsCheckingIcLora] = useState(false)
  const [isDownloadingIcLora, setIsDownloadingIcLora] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState<IcLoraDownloadProgress | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [extractError, setExtractError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const requiredModelIds: IcLoraModelId[] = modelType === 'motion_track'
    ? ['ic_lora_motion_track']
    : conditioningType === 'depth'
      ? ['ic_lora', 'depth_processor']
      : conditioningType === 'pose'
        ? ['ic_lora', 'person_detector', 'pose_processor']
        : ['ic_lora']
  const requiredModelIdsKey = requiredModelIds.join('|')
  const icLoraReady = requiredModelIds.every(id => icModelDownloaded[id])

  useEffect(() => {
    if (resetKey === undefined) return
    setInputVideoUrl(initialVideoUrl || null)
    setInputVideoPath(initialVideoPath || null)
    setInputImageUrl(initialImageUrl || null)
    setInputImagePath(initialImagePath || null)
    setInputTime(0)
    setInternalModelType('union')
    setInternalCondType('canny')
    setInternalCondStrength(1.0)
    onModelTypeChange?.('union')
    onConditioningTypeChange?.('canny')
    onConditioningStrengthChange?.(1.0)
    setConditioningPreview(null)
    setExtractError(null)
  }, [resetKey, initialVideoUrl, initialVideoPath, initialImageUrl, initialImagePath]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const allowed = getConditioningTypesForModel(modelType)
    if (allowed.some(option => option.value === conditioningType)) return
    const fallback = allowed[0]?.value ?? 'canny'
    if (conditioningTypeProp === undefined) {
      setInternalCondType(fallback)
    }
    onConditioningTypeChange?.(fallback)
  }, [modelType, conditioningType, conditioningTypeProp, onConditioningTypeChange])

  useEffect(() => {
    const ready = !!inputVideoPath && icLoraReady
    onChange?.({
      videoUrl: inputVideoUrl,
      videoPath: inputVideoPath,
      imageUrl: inputImageUrl,
      imagePath: inputImagePath,
      modelType,
      conditioningType,
      conditioningStrength,
      resolution,
      aspectRatio,
      duration,
      ready,
    })
  }, [inputVideoUrl, inputVideoPath, inputImageUrl, inputImagePath, modelType, conditioningType, conditioningStrength, resolution, aspectRatio, duration, icLoraReady, onChange])

  const checkIcLoraAvailability = useCallback(async () => {
    setIsCheckingIcLora(true)
    try {
      const statusResponse = await backendFetch('/api/models/status')
      if (!statusResponse.ok) {
        setDownloadError(`Failed to fetch model status (${statusResponse.status})`)
        return
      }

      const statusPayload = await statusResponse.json() as ModelsStatusResponse
      const nextStatus: Record<IcLoraModelId, boolean> = { ...EMPTY_IC_MODEL_STATUS }
      IC_LORA_MODEL_IDS.forEach(modelId => {
        nextStatus[modelId] = statusPayload.models.some(model => model.id === modelId && model.downloaded)
      })
      setIcModelDownloaded(nextStatus)
      const isReady = requiredModelIds.every(modelId => nextStatus[modelId])

      if (isReady) {
        setIsDownloadingIcLora(false)
        setDownloadProgress(null)
        setDownloadError(null)
        setDownloadSessionId(null)
      }
    } catch (e) {
      logger.warn(`Failed to fetch IC-LoRA model status: ${e}`)
      setDownloadError((e as Error).message)
    } finally {
      setIsCheckingIcLora(false)
    }
  }, [requiredModelIdsKey])

  useEffect(() => {
    void checkIcLoraAvailability()
  }, [checkIcLoraAvailability])

  useEffect(() => {
    if (icLoraReady || !isDownloadingIcLora || !downloadSessionId) return

    const pollProgress = async () => {
      try {
        const progressResponse = await backendFetch(`/api/models/download/progress?sessionId=${downloadSessionId}`)
        if (!progressResponse.ok) {
          return
        }

        const progressPayload = await progressResponse.json() as IcLoraDownloadProgress
        setDownloadProgress(progressPayload)

        if (progressPayload.status === 'error') {
          setIsDownloadingIcLora(false)
          setDownloadError(progressPayload.error || 'Model download failed')
          return
        }

        if (progressPayload.status === 'complete') {
          setIsDownloadingIcLora(false)
          await checkIcLoraAvailability()
        }
      } catch (e) {
        logger.warn(`Failed polling IC-LoRA download progress: ${e}`)
      }
    }

    void pollProgress()
    const interval = setInterval(() => { void pollProgress() }, 1000)
    return () => clearInterval(interval)
  }, [icLoraReady, isDownloadingIcLora, downloadSessionId, checkIcLoraAvailability, requiredModelIdsKey])

  const handleDownloadIcLora = useCallback(async () => {
    if (isDownloadingIcLora) return
    setDownloadError(null)

    try {
      const response = await backendFetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelTypes: requiredModelIds }),
      })

      const payload = await response.json().catch(() => ({})) as ModelDownloadStartResponse
      if (!response.ok) {
        const errorMessage = payload.error || payload.message || `Download request failed (${response.status})`
        setDownloadError(errorMessage)
        return
      }

      if (payload.status === 'started') {
        if (payload.sessionId) {
          setDownloadSessionId(payload.sessionId)
        }
        setIsDownloadingIcLora(true)
        return
      }

      setDownloadError('Unexpected response while starting IC-LoRA download')
    } catch (e) {
      logger.warn(`Failed to start IC-LoRA download: ${e}`)
      setDownloadError((e as Error).message)
    }
  }, [isDownloadingIcLora, requiredModelIdsKey])

  const isExtractingRef = useRef(false)
  const extractConditioning = useCallback(async () => {
    if (!inputVideoPath || isExtractingRef.current || !icLoraReady) return
    isExtractingRef.current = true
    setIsExtracting(true)
    setExtractError(null)
    try {
      const response = await backendFetch('/api/ic-lora/extract-conditioning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: inputVideoPath,
          model_type: modelType,
          conditioning_type: conditioningType,
          frame_time: inputTime,
        }),
      })
      if (response.ok) {
        const payload = await response.json()
        setConditioningPreview(payload.conditioning)
        return
      }
      const payload = await response.json().catch(() => ({} as { error?: string }))
      setExtractError(payload.error || `Failed to extract conditioning (${response.status})`)
    } catch (e) {
      logger.warn(`Failed to extract conditioning: ${e}`)
      setExtractError((e as Error).message)
    } finally {
      isExtractingRef.current = false
      setIsExtracting(false)
    }
  }, [inputVideoPath, modelType, conditioningType, inputTime, icLoraReady])

  const extractTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (!inputVideoPath || !icLoraReady) return
    if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    extractTimerRef.current = setTimeout(() => {
      void extractConditioning()
    }, 300)
    return () => {
      if (extractTimerRef.current) clearTimeout(extractTimerRef.current)
    }
  }, [inputTime, conditioningType, inputVideoPath, icLoraReady, extractConditioning])

  useEffect(() => {
    const video = inputVideoRef.current
    if (!video) return
    const onTime = () => setInputTime(video.currentTime)
    const onSeeked = () => setInputTime(video.currentTime)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('seeked', onSeeked)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('seeked', onSeeked)
    }
  }, [inputVideoUrl, icLoraReady, isCheckingIcLora])

  const handleBrowse = useCallback(async () => {
    try {
      const selected = await selectLocalFile({
        title: 'Select Driving Video',
        extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'],
        accept: 'video/*,.mp4,.mov,.avi,.webm,.mkv',
        kind: 'video',
      })
      if (!selected) return
      setInputVideoPath(selected.path)
      setInputVideoUrl(selected.url)
      setConditioningPreview(null)
      setExtractError(null)
    } catch (error) {
      setExtractError(error instanceof Error ? error.message : 'Failed to load video')
    }
  }, [])

  const handleClear = useCallback(() => {
    setInputVideoPath(null)
    setInputVideoUrl(null)
    setInputImagePath(null)
    setInputImageUrl(null)
    setInputTime(0)
    setConditioningPreview(null)
    setExtractError(null)
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; url?: string; path?: string }
        if (asset.type === 'video' && asset.url) {
          const path = asset.path || fileUrlToPath(asset.url) || null
          setInputVideoUrl(asset.url)
          setInputVideoPath(path)
          setConditioningPreview(null)
          setExtractError(null)
          return
        }
      } catch {
        // fall through
      }
    }

    const file = e.dataTransfer.files?.[0]
    if (file) {
      try {
        const persisted = await persistDroppedFile(file, 'video')
        setInputVideoPath(persisted.path)
        setInputVideoUrl(persisted.url)
        setConditioningPreview(null)
        setExtractError(null)
      } catch (error) {
        setExtractError(error instanceof Error ? error.message : 'Failed to load dropped video')
      }
    }
  }, [])

  const handleBrowseImage = useCallback(async () => {
    try {
      const selected = await selectLocalFile({
        title: 'Select Starting Frame',
        extensions: ['png', 'jpg', 'jpeg', 'webp'],
        accept: 'image/*,.png,.jpg,.jpeg,.webp',
        kind: 'image',
      })
      if (!selected) return
      setInputImagePath(selected.path)
      setInputImageUrl(selected.url)
    } catch (error) {
      setExtractError(error instanceof Error ? error.message : 'Failed to load image')
    }
  }, [])

  const handleClearImage = useCallback(() => {
    setInputImagePath(null)
    setInputImageUrl(null)
  }, [])

  const showDownloadGate = isCheckingIcLora || !icLoraReady
  const gateItems = requiredModelIds.map(modelId => {
    const downloaded = icModelDownloaded[modelId]
    const isCompleted = downloadProgress?.completed_files?.includes(modelId) ?? false
    const isCurrentDownload = isDownloadingIcLora && downloadProgress?.current_downloading_file === modelId
    const progress = downloaded ? 100 : (isCompleted ? 100 : (isCurrentDownload ? (downloadProgress?.current_file_progress ?? 0) : 0))
    const status = downloaded ? 'Ready' : (isCompleted ? 'Complete' : (isCurrentDownload ? 'Downloading' : 'Missing'))
    return { id: modelId, label: IC_LORA_MODEL_LABELS[modelId], downloaded, progress, status }
  })

  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden flex flex-col ${fillHeight ? 'h-full min-h-0' : ''}`}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-amber-400" />
          <span className="text-sm font-semibold text-white">IC-LoRA / Style Transfer</span>
        </div>
        <div className="flex items-center gap-2">
          {inputVideoUrl && (
            <>
              <button
                onClick={handleClear}
                className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                title="Clear video"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={handleBrowse}
                className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
                title="Replace video"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {showDownloadGate ? (
        <div className="flex-1 flex items-center justify-center p-6 min-h-0 overflow-y-auto">
          <div className="w-full max-w-xl rounded-xl border border-zinc-700 bg-zinc-800/60 p-6">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg bg-blue-600/20 flex items-center justify-center mt-0.5">
                <Download className="h-4 w-4 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold text-white">Download Required: IC-LoRA Resources</h3>
                <p className="text-xs text-zinc-400 mt-1">
                  The selected IC-LoRA mode is locked until its required local models are available.
                </p>
              </div>
            </div>

            <div className="mt-5 space-y-3">
              {isCheckingIcLora ? (
                <div className="flex items-center gap-2 text-xs text-zinc-300">
                  <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                  Checking model availability...
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    {gateItems.map(item => (
                      <div key={item.id} className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2">
                        <div className="flex items-center justify-between text-[11px] mb-1.5">
                          <span className="text-zinc-300">{item.label}</span>
                          <span className={item.downloaded ? 'text-blue-400' : 'text-zinc-500'}>
                            {item.status}
                          </span>
                        </div>
                        <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className="h-full transition-all duration-300 bg-blue-500"
                            style={{ width: `${item.progress}%` }}
                          />
                        </div>
                        <div className="mt-1 text-[10px] text-zinc-500">{item.progress}%</div>
                      </div>
                    ))}
                  </div>
                  {downloadError && (
                    <div className="text-[11px] text-red-400">{downloadError}</div>
                  )}
                  <div className="flex items-center gap-2 pt-1">
                    <button
                      onClick={handleDownloadIcLora}
                      disabled={isDownloadingIcLora}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isDownloadingIcLora ? (
                        <>
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Downloading...
                        </>
                      ) : (
                        <>
                          <Download className="h-3 w-3" />
                          {downloadError ? 'Retry Download' : 'Download Models'}
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => { void checkIcLoraAvailability() }}
                      disabled={isCheckingIcLora}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-600 text-zinc-300 hover:text-white hover:border-zinc-500 text-xs transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`h-3 w-3 ${isCheckingIcLora ? 'animate-spin' : ''}`} />
                      Refresh
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0 overflow-hidden">
          <div className={`${showInlineOutputPreview ? 'flex-1 border-r border-zinc-800' : 'w-1/2 border-r border-zinc-800'} flex flex-col min-w-0`}>
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider shrink-0">Control Video</span>
              {inputVideoPath && (
                <span className="text-[10px] text-zinc-500 truncate min-w-0">
                  {inputVideoPath.split(/[\\/]/).pop()}
                </span>
              )}
              <button
                onClick={handleBrowse}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors shrink-0"
              >
                <Upload className="h-3 w-3" />
                Import
              </button>
            </div>
            <div
              className={`flex-1 min-h-0 bg-black flex items-center justify-center relative ${!inputVideoUrl ? 'border-2 border-dashed border-zinc-700 m-3 rounded-lg' : ''} ${isDragOver ? 'border-blue-500 bg-blue-500/10' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => { void handleDrop(e) }}
            >
              {inputVideoUrl ? (
                <video
                  ref={inputVideoRef}
                  src={inputVideoUrl}
                  className="w-full h-full object-contain"
                  controls
                />
              ) : (
                <div className="text-center p-4">
                  <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-2">
                    <Film className="h-6 w-6 text-zinc-600" />
                  </div>
                  <p className="text-zinc-400 text-xs">Drop or import a control video</p>
                  {modelType === 'motion_track' && (
                    <p className="text-[10px] text-zinc-500 mt-2 max-w-[220px] mx-auto">
                      Motion Track expects a trajectory-overlay video, not a plain reference clip.
                    </p>
                  )}
                  <button
                    onClick={handleBrowse}
                    className="mt-2 px-3 py-1.5 text-[10px] text-blue-400 border border-blue-500/30 rounded-lg hover:bg-blue-600/10 transition-colors"
                  >
                    Import Video
                  </button>
                </div>
              )}
            </div>
            <div className="border-t border-zinc-800 p-3 bg-zinc-950/60">
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Starting Frame</span>
                <div className="flex items-center gap-1">
                  {inputImageUrl && (
                    <button
                      onClick={handleClearImage}
                      className="p-1 rounded text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
                      title="Clear starting frame"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                  <button
                    onClick={handleBrowseImage}
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors shrink-0"
                  >
                    <Upload className="h-3 w-3" />
                    {inputImageUrl ? 'Replace' : 'Import'}
                  </button>
                </div>
              </div>
              <div className="h-24 rounded-lg border border-zinc-800 bg-black flex items-center justify-center overflow-hidden">
                {inputImageUrl ? (
                  <img src={inputImageUrl} alt="Starting frame" className="w-full h-full object-contain" />
                ) : (
                  <p className="text-zinc-600 text-xs text-center px-3">
                    Optional image reference for the first frame / image-to-video guidance
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className={`${showInlineOutputPreview ? 'flex-1' : 'w-1/2'} flex flex-col min-w-0`}>
            <div className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-2">
              <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Conditioning</span>
              <button
                onClick={() => { void extractConditioning() }}
                disabled={!inputVideoPath || isExtracting}
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`h-3 w-3 ${isExtracting ? 'animate-spin' : ''}`} />
              </button>
            </div>
            <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
              {isExtracting && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                  <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                </div>
              )}
              {conditioningPreview ? (
                <img src={conditioningPreview} alt="Conditioning preview" className="w-full h-full object-contain" />
              ) : (
                <div className="text-center p-4">
                  <p className="text-zinc-600 text-xs">
                    {inputVideoUrl ? 'Scrub the input video to see conditioning preview' : 'Import a video to preview conditioning'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {showInlineOutputPreview && (
            <div className="flex-1 flex flex-col border-l border-zinc-800 min-w-0">
              <div className="px-3 py-2 border-b border-zinc-800 flex items-center">
                <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">Output</span>
              </div>
              <div className="flex-1 bg-black flex items-center justify-center min-h-0 relative">
                {outputVideoUrl ? (
                  <video
                    src={outputVideoUrl}
                    className="w-full h-full object-contain"
                    controls
                  />
                ) : isProcessing ? (
                  <div className="text-center p-4">
                    <Loader2 className="h-6 w-6 text-blue-400 animate-spin mx-auto mb-2" />
                    <p className="text-zinc-400 text-xs">{processingStatus || 'Generating...'}</p>
                  </div>
                ) : (
                  <div className="text-center p-4">
                    <p className="text-zinc-600 text-xs">Output video will appear here</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {!showDownloadGate && (
        <div className="grid grid-cols-3 gap-3 px-4 py-3 border-t border-zinc-800 flex-shrink-0">
          <Select
            label="Duration"
            value={duration == null ? 'full' : String(duration)}
            onChange={(e) => onDurationChange?.(e.target.value === 'full' ? null : Number(e.target.value))}
          >
            <option value="full">Full Ref</option>
            <option value="1">1 sec</option>
            <option value="2">2 sec</option>
            <option value="3">3 sec</option>
            <option value="4">4 sec</option>
            <option value="5">5 sec</option>
            <option value="6">6 sec</option>
            <option value="8">8 sec</option>
            <option value="10">10 sec</option>
            <option value="12">12 sec</option>
            <option value="15">15 sec</option>
            <option value="20">20 sec</option>
          </Select>
          <Select
            label="Resolution"
            value={resolution}
            onChange={(e) => onResolutionChange?.(e.target.value as '540p' | '720p' | '1080p')}
          >
            <option value="540p">540p</option>
            <option value="720p">720p</option>
            <option value="1080p">1080p</option>
          </Select>
          <Select
            label="Aspect Ratio"
            value={aspectRatio}
            onChange={(e) => onAspectRatioChange?.(e.target.value as '16:9' | '9:16')}
          >
            <option value="16:9">16:9 Landscape</option>
            <option value="9:16">9:16 Portrait</option>
          </Select>
        </div>
      )}

      {extractError && (
        <div className="px-4 py-3 border-t border-zinc-800 flex-shrink-0">
          <div className="flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
            <span>{extractError}</span>
          </div>
        </div>
      )}
    </div>
  )
}
