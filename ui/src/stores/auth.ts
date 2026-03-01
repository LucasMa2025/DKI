import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginRequest } from '@/types'
import { api } from '@/services/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const token = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  // 记录上一次登录的用户 ID，用于检测用户切换
  const _lastUserId = ref<string | null>(null)
  
  const isAuthenticated = computed(() => !!token.value && !!user.value)
  
  /**
   * 清除其他 store 中与用户绑定的数据
   * 防止用户切换后看到上一个用户的 session / messages
   */
  function _clearUserBoundStores() {
    try {
      // 延迟导入避免循环依赖
      const { useChatStore } = require('./chat')
      const chatStore = useChatStore()
      chatStore.resetState()
    } catch {
      // chat store 可能尚未初始化
    }
  }
  
  async function login(credentials: LoginRequest) {
    loading.value = true
    error.value = null
    
    try {
      const response = await api.auth.login(credentials)
      
      // 检测用户切换: 如果新登录的用户与上次不同，清除旧数据
      const newUserId = response.user?.id
      if (_lastUserId.value && _lastUserId.value !== newUserId) {
        _clearUserBoundStores()
      }
      
      token.value = response.token
      user.value = response.user
      _lastUserId.value = newUserId || null
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Login failed'
      return false
    } finally {
      loading.value = false
    }
  }
  
  async function logout() {
    try {
      await api.auth.logout()
    } catch {
      // Ignore logout errors
    } finally {
      // 清除所有用户绑定的数据
      _clearUserBoundStores()
      user.value = null
      token.value = null
      _lastUserId.value = null
    }
  }
  
  async function refreshUser() {
    if (!token.value) return
    
    try {
      const userData = await api.auth.getCurrentUser()
      user.value = userData
    } catch {
      // Token might be invalid
      await logout()
    }
  }
  
  return {
    user,
    token,
    loading,
    error,
    isAuthenticated,
    login,
    logout,
    refreshUser,
  }
}, {
  persist: {
    paths: ['token', 'user'],
  },
})
