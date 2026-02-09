import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ChatMessage, Session, ChatRequest } from '@/types'
import { api } from '@/services/api'
import { useAuthStore } from './auth'
import { useSettingsStore } from './settings'

export const useChatStore = defineStore('chat', () => {
  const sessions = ref<Session[]>([])
  const currentSessionId = ref<string | null>(null)
  const messages = ref<ChatMessage[]>([])
  const loading = ref(false)
  const streaming = ref(false)
  const error = ref<string | null>(null)
  
  const currentSession = computed(() => {
    return sessions.value.find(s => s.id === currentSessionId.value) || null
  })
  
  // Load sessions
  async function loadSessions() {
    try {
      sessions.value = await api.sessions.list()
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }
  
  // Create new session
  async function createSession(title?: string): Promise<Session | null> {
    try {
      const session = await api.sessions.create(title || '新对话')
      sessions.value.unshift(session)
      return session
    } catch (e) {
      error.value = '创建会话失败'
      return null
    }
  }
  
  // Select session
  async function selectSession(sessionId: string) {
    if (currentSessionId.value === sessionId) return
    
    currentSessionId.value = sessionId
    messages.value = []
    
    try {
      loading.value = true
      messages.value = await api.sessions.getMessages(sessionId)
    } catch (e) {
      error.value = '加载消息失败'
    } finally {
      loading.value = false
    }
  }
  
  // Delete session
  async function deleteSession(sessionId: string) {
    try {
      await api.sessions.delete(sessionId)
      sessions.value = sessions.value.filter(s => s.id !== sessionId)
      
      if (currentSessionId.value === sessionId) {
        currentSessionId.value = sessions.value[0]?.id || null
        messages.value = []
      }
    } catch (e) {
      error.value = '删除会话失败'
    }
  }
  
  // Rename session
  async function renameSession(sessionId: string, title: string) {
    try {
      await api.sessions.update(sessionId, { title })
      const session = sessions.value.find(s => s.id === sessionId)
      if (session) {
        session.title = title
      }
    } catch (e) {
      error.value = '重命名失败'
    }
  }
  
  // Send message
  // 修正: 只传递 user_id 和原始输入，移除 prompt 拼接逻辑
  // DKI 会自动通过适配器读取用户偏好和历史消息进行注入
  async function sendMessage(content: string) {
    const authStore = useAuthStore()
    const settingsStore = useSettingsStore()
    
    if (!content.trim()) return
    
    // Ensure we have a session
    if (!currentSessionId.value) {
      const session = await createSession()
      if (!session) return
      currentSessionId.value = session.id
    }
    
    // Add user message to local display
    const userMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      sessionId: currentSessionId.value,
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString(),
    }
    messages.value.push(userMessage)
    
    // 修正: 简化请求，只传递必要信息
    // - user_id: 用户标识 (DKI 用于读取偏好和历史)
    // - session_id: 会话标识 (DKI 用于读取会话历史)
    // - query: 原始用户输入 (不含任何 prompt 构造)
    // DKI 会自动:
    // 1. 通过适配器读取用户偏好 → K/V 注入
    // 2. 通过适配器检索相关历史 → 后缀提示词
    const request: ChatRequest = {
      // 原始用户输入，不拼接任何历史消息或 prompt
      query: content.trim(),
      // 用户标识 - DKI 用于读取偏好和历史
      dkiUserId: authStore.user?.id,
      // 会话标识 - DKI 用于读取会话历史
      dkiSessionId: currentSessionId.value,
      // 可选参数
      model: settingsStore.defaultModel,
      temperature: settingsStore.temperature,
      maxTokens: settingsStore.maxTokens,
      stream: false,
    }
    
    // Add placeholder for assistant message
    const assistantMessage: ChatMessage = {
      id: `temp-assistant-${Date.now()}`,
      sessionId: currentSessionId.value,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    }
    messages.value.push(assistantMessage)
    
    try {
      loading.value = true
      error.value = null
      
      const response = await api.chat.send(request)
      
      // Update assistant message
      const lastMessage = messages.value[messages.value.length - 1]
      if (lastMessage.role === 'assistant') {
        lastMessage.id = response.id
        lastMessage.content = response.choices[0]?.message.content || ''
        lastMessage.dkiMetadata = response.dkiMetadata
      }
      
      // Update session preview
      const session = sessions.value.find(s => s.id === currentSessionId.value)
      if (session) {
        session.preview = content.slice(0, 50)
        session.messageCount = messages.value.length
        session.updatedAt = new Date().toISOString()
      }
      
    } catch (e) {
      error.value = e instanceof Error ? e.message : '发送消息失败'
      // Remove the placeholder message on error
      messages.value.pop()
    } finally {
      loading.value = false
      streaming.value = false
    }
  }
  
  // Clear current session messages
  function clearMessages() {
    messages.value = []
  }
  
  // Clear error
  function clearError() {
    error.value = null
  }
  
  return {
    sessions,
    currentSessionId,
    messages,
    loading,
    streaming,
    error,
    currentSession,
    loadSessions,
    createSession,
    selectSession,
    deleteSession,
    renameSession,
    sendMessage,
    clearMessages,
    clearError,
  }
})
