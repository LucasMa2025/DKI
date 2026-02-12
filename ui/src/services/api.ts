import axios, { AxiosInstance, AxiosError } from 'axios'
import type {
  User,
  LoginRequest,
  LoginResponse,
  ChatRequest,
  ChatResponse,
  Session,
  ChatMessage,
  UserPreference,
  SystemStats,
} from '@/types'
import config from '@/config'

// Create axios instance
const http: AxiosInstance = axios.create({
  baseURL: config.api.baseUrl,
  timeout: config.api.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
http.interceptors.request.use(
  (config) => {
    // Get token from localStorage
    const authData = localStorage.getItem('auth')
    if (authData) {
      try {
        const { token } = JSON.parse(authData)
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
      } catch {
        // Ignore parse errors
      }
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor
http.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Clear auth data and redirect to login
      localStorage.removeItem('auth')
      window.location.href = '/login'
    }
    
    const message = (error.response?.data as { detail?: string })?.detail || error.message
    return Promise.reject(new Error(message))
  }
)

// API methods
export const api = {
  // Auth
  auth: {
    async login(credentials: LoginRequest): Promise<LoginResponse> {
      return http.post('/auth/login', credentials)
    },
    
    async logout(): Promise<void> {
      return http.post('/auth/logout')
    },
    
    async getCurrentUser(): Promise<User> {
      return http.get('/auth/me')
    },
    
    async register(data: { username: string; password: string; email?: string }): Promise<User> {
      return http.post('/auth/register', data)
    },
  },
  
  // Chat
  // 修正: 简化 API 调用，只传递 user_id 和原始输入
  // DKI 会自动处理偏好读取、历史检索和注入
  chat: {
    async send(request: ChatRequest): Promise<ChatResponse> {
      // 修正: 使用 DKI 插件 API
      // 只传递:
      // - query: 原始用户输入 (不含任何 prompt 构造)
      // - user_id: 用户标识 (DKI 用于读取偏好和历史)
      // - session_id: 会话标识 (DKI 用于读取会话历史)
      return http.post('/v1/dki/chat', {
        // 原始用户输入，不拼接任何历史或 prompt
        query: request.query,
        // 用户标识 - DKI 用于读取偏好和历史
        user_id: request.dkiUserId,
        // 会话标识 - DKI 用于读取会话历史
        session_id: request.dkiSessionId,
        // 可选参数
        model: request.model,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
      })
    },
    
    // Streaming chat (returns EventSource)
    createStream(request: ChatRequest): EventSource {
      const params = new URLSearchParams({
        query: request.query || '',
        user_id: request.dkiUserId || '',
        session_id: request.dkiSessionId || '',
        model: request.model || '',
        temperature: String(request.temperature || 0.7),
        max_tokens: String(request.maxTokens || 2048),
      })
      
      return new EventSource(`${config.api.baseUrl}/v1/dki/chat/stream?${params}`)
    },
  },
  
  // Sessions
  sessions: {
    async list(): Promise<Session[]> {
      return http.get('/sessions')
    },
    
    async get(id: string): Promise<Session> {
      return http.get(`/sessions/${id}`)
    },
    
    async create(title: string): Promise<Session> {
      return http.post('/sessions', { title })
    },
    
    async update(id: string, data: Partial<Session>): Promise<Session> {
      return http.patch(`/sessions/${id}`, data)
    },
    
    async delete(id: string): Promise<void> {
      return http.delete(`/sessions/${id}`)
    },
    
    async getMessages(sessionId: string): Promise<ChatMessage[]> {
      return http.get(`/sessions/${sessionId}/messages`)
    },
  },
  
  // Preferences
  preferences: {
    async list(userId: string): Promise<UserPreference[]> {
      return http.get(`/preferences`, { params: { user_id: userId } })
    },
    
    async get(id: string): Promise<UserPreference> {
      return http.get(`/preferences/${id}`)
    },
    
    async create(preference: Omit<UserPreference, 'id' | 'createdAt' | 'updatedAt'>): Promise<UserPreference> {
      return http.post('/preferences', {
        user_id: preference.userId,
        preference_text: preference.preferenceText,
        preference_type: preference.preferenceType,
        priority: preference.priority,
        category: preference.category,
        metadata: preference.metadata,
        is_active: preference.isActive,
      })
    },
    
    async update(id: string, updates: Partial<UserPreference>): Promise<UserPreference> {
      return http.patch(`/preferences/${id}`, {
        preference_text: updates.preferenceText,
        preference_type: updates.preferenceType,
        priority: updates.priority,
        category: updates.category,
        metadata: updates.metadata,
        is_active: updates.isActive,
      })
    },
    
    async delete(id: string): Promise<void> {
      return http.delete(`/preferences/${id}`)
    },
  },
  
  // Stats
  stats: {
    async getSystemStats(): Promise<SystemStats> {
      return http.get('/stats')
    },
    
    async getDKIStats(): Promise<SystemStats['dkiStats']> {
      return http.get('/stats/dki')
    },
    
    async getCacheStats(): Promise<SystemStats['cacheStats']> {
      return http.get('/stats/cache')
    },
  },
  
  // Health
  health: {
    async check(): Promise<{ status: string; version: string }> {
      return http.get('/health')
    },
  },
  
  // Visualization
  visualization: {
    async getLatest(): Promise<any> {
      return http.get('/v1/dki/visualization/latest')
    },
    
    async getHistory(page: number = 1, pageSize: number = 20): Promise<{ items: any[]; total: number; page: number; page_size: number }> {
      return http.get('/v1/dki/visualization/history', { params: { page, page_size: pageSize } })
    },
    
    async getDetail(requestId: string): Promise<any> {
      return http.get(`/v1/dki/visualization/detail/${requestId}`)
    },
    
    async getFlowDiagram(): Promise<any> {
      return http.get('/v1/dki/visualization/flow-diagram')
    },
    
    async clearHistory(): Promise<{ message: string; success: boolean }> {
      return http.delete('/v1/dki/visualization/history')
    },
  },
}

export default api
