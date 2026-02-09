// User types
export interface User {
  id: string
  username: string
  email?: string
  avatar?: string
  createdAt?: string
}

// Auth types
export interface LoginRequest {
  username: string
  password: string
  remember?: boolean
}

export interface LoginResponse {
  token: string
  user: User
}

// Chat types
export interface ChatMessage {
  id: string
  sessionId: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  dkiMetadata?: DKIMetadata
}

export interface DKIMetadata {
  injectionEnabled: boolean
  alpha?: number
  memoriesUsed: number
  preferenceTokens: number
  historyTokens: number
  cacheHit: boolean
  cacheTier?: string
  latencyMs: number
  gatingDecision?: GatingDecision
}

export interface GatingDecision {
  shouldInject: boolean
  relevanceScore: number
  entropy: number
  reasoning: string
}

export interface ChatRequest {
  model?: string
  messages: Array<{ role: string; content: string }>
  temperature?: number
  maxTokens?: number
  stream?: boolean
  dkiEnabled?: boolean
  dkiUserId?: string
  dkiSessionId?: string
  dkiForceAlpha?: number
  dkiUseHybrid?: boolean
}

export interface ChatResponse {
  id: string
  object: string
  created: number
  model: string
  choices: Array<{
    index: number
    message: { role: string; content: string }
    finishReason: string
  }>
  usage: {
    promptTokens: number
    completionTokens: number
    totalTokens: number
  }
  dkiMetadata?: DKIMetadata
}

// Session types
export interface Session {
  id: string
  title: string
  userId?: string
  messageCount: number
  createdAt: string
  updatedAt: string
  preview?: string
}

// Preference types
export interface UserPreference {
  id?: string
  userId: string
  preferenceText: string
  preferenceType: string
  priority: number
  category?: string
  metadata?: Record<string, unknown>
  createdAt?: string
  updatedAt?: string
  expiresAt?: string
  isActive: boolean
}

// Stats types
export interface SystemStats {
  dkiStats: {
    totalRequests: number
    l1Hits: number
    l2Hits: number
    l3Computes: number
    avgAlpha: number
    injectionRate: number
  }
  cacheStats: {
    l1Size: number
    l1MaxSize: number
    l1HitRate: number
    l2HitRate: number
  }
  adapterStats: {
    type: string
    connected: boolean
  }
  uptimeSeconds: number
}

// Settings types
export interface AppSettings {
  language: 'zh-CN' | 'en-US'
  theme: 'light' | 'dark' | 'system'
  fontSize: number
  sendOnEnter: boolean
  showTimestamps: boolean
  compactMode: boolean
}

export interface ModelSettings {
  defaultModel: string
  temperature: number
  maxTokens: number
  topP: number
}

export interface DKISettings {
  enabled: boolean
  defaultAlpha: number
  useHybrid: boolean
  debugMode: boolean
}

export interface APISettings {
  baseUrl: string
  apiKey: string
  timeout: number
}

// Config for stats page auth
export interface StatsAuthConfig {
  enabled: boolean
  password: string
}
