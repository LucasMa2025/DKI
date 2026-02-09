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
  chat: {
    async send(request: ChatRequest): Promise<ChatResponse> {
      return http.post('/v1/chat/completions', {
        model: request.model,
        messages: request.messages,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        stream: request.stream,
        dki_user_id: request.dkiUserId,
        dki_session_id: request.dkiSessionId,
        dki_force_alpha: request.dkiForceAlpha,
      })
    },
    
    // Streaming chat (returns EventSource)
    createStream(request: ChatRequest): EventSource {
      const params = new URLSearchParams({
        model: request.model || '',
        temperature: String(request.temperature || 0.7),
        max_tokens: String(request.maxTokens || 2048),
        dki_user_id: request.dkiUserId || '',
        dki_session_id: request.dkiSessionId || '',
      })
      
      return new EventSource(`${config.api.baseUrl}/v1/chat/stream?${params}`)
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
}

export default api
