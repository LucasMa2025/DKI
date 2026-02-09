import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { User, LoginRequest } from '@/types'
import { api } from '@/services/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const token = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  const isAuthenticated = computed(() => !!token.value && !!user.value)
  
  async function login(credentials: LoginRequest) {
    loading.value = true
    error.value = null
    
    try {
      const response = await api.auth.login(credentials)
      token.value = response.token
      user.value = response.user
      return true
    } catch (e) {
      error.value = e instanceof Error ? e.message : '登录失败'
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
      user.value = null
      token.value = null
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
