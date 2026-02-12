<template>
  <div class="main-layout">
    <!-- Sidebar -->
    <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
      <div class="sidebar-header">
        <div class="logo" v-if="!sidebarCollapsed">
          <img src="/logo.svg" alt="DKI" class="logo-icon" />
          <span class="logo-text">DKI Chat</span>
        </div>
        <el-button
          class="collapse-btn"
          :icon="sidebarCollapsed ? 'Expand' : 'Fold'"
          text
          @click="toggleSidebar"
        />
      </div>
      
      <div class="sidebar-content">
        <!-- New Chat Button -->
        <el-button
          type="primary"
          class="new-chat-btn"
          :icon="Plus"
          @click="handleNewChat"
        >
          <span v-if="!sidebarCollapsed">新对话</span>
        </el-button>
        
        <!-- Navigation -->
        <nav class="nav-menu">
          <router-link
            v-for="item in navItems"
            :key="item.path"
            :to="item.path"
            class="nav-item"
            :class="{ active: isActive(item.path) }"
          >
            <el-icon><component :is="item.icon" /></el-icon>
            <span v-if="!sidebarCollapsed">{{ item.label }}</span>
          </router-link>
        </nav>
        
        <!-- Session List -->
        <div class="session-list" v-if="!sidebarCollapsed">
          <div class="session-list-header">
            <span>最近对话</span>
          </div>
          <div class="session-items">
            <div
              v-for="session in recentSessions"
              :key="session.id"
              class="session-item"
              :class="{ active: session.id === currentSessionId }"
              @click="selectSession(session.id)"
            >
              <el-icon><ChatDotRound /></el-icon>
              <span class="session-title">{{ session.title }}</span>
              <el-dropdown trigger="click" @command="handleSessionCommand($event, session.id)">
                <el-button class="session-menu-btn" :icon="MoreFilled" text size="small" />
                <template #dropdown>
                  <el-dropdown-menu>
                    <el-dropdown-item command="rename">
                      <el-icon><Edit /></el-icon>重命名
                    </el-dropdown-item>
                    <el-dropdown-item command="delete" divided>
                      <el-icon><Delete /></el-icon>删除
                    </el-dropdown-item>
                  </el-dropdown-menu>
                </template>
              </el-dropdown>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Sidebar Footer -->
      <div class="sidebar-footer">
        <div class="user-info" v-if="!sidebarCollapsed">
          <el-avatar :size="32" :src="user?.avatar">
            {{ user?.username?.charAt(0).toUpperCase() }}
          </el-avatar>
          <span class="username">{{ user?.username }}</span>
        </div>
        <el-dropdown trigger="click" @command="handleUserCommand">
          <el-button :icon="Setting" text />
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="settings">
                <el-icon><Setting /></el-icon>设置
              </el-dropdown-item>
              <el-dropdown-item command="theme">
                <el-icon><Moon v-if="theme === 'light'" /><Sunny v-else /></el-icon>
                {{ theme === 'light' ? '深色模式' : '浅色模式' }}
              </el-dropdown-item>
              <el-dropdown-item command="logout" divided>
                <el-icon><SwitchButton /></el-icon>退出登录
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </aside>
    
    <!-- Main Content -->
    <main class="main-content">
      <router-view />
    </main>
    
    <!-- Settings Dialog -->
    <SettingsDialog v-model="showSettings" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  Plus,
  ChatDotRound,
  Document,
  DataAnalysis,
  Setting,
  Moon,
  Sunny,
  SwitchButton,
  MoreFilled,
  Edit,
  Delete,
  View,
} from '@element-plus/icons-vue'
import { ElMessageBox, ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { useSettingsStore } from '@/stores/settings'
import SettingsDialog from '@/components/SettingsDialog.vue'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const chatStore = useChatStore()
const settingsStore = useSettingsStore()

const sidebarCollapsed = ref(false)
const showSettings = ref(false)

const user = computed(() => authStore.user)
const theme = computed(() => settingsStore.theme)
const recentSessions = computed(() => chatStore.sessions.slice(0, 10))
const currentSessionId = computed(() => chatStore.currentSessionId)

const navItems = [
  { path: '/', icon: ChatDotRound, label: '聊天' },
  { path: '/sessions', icon: Document, label: '会话管理' },
  { path: '/preferences', icon: Setting, label: '偏好设置' },
  { path: '/visualization', icon: View, label: '注入可视化' },
  { path: '/stats', icon: DataAnalysis, label: '系统统计' },
]

function isActive(path: string) {
  if (path === '/') {
    return route.path === '/'
  }
  return route.path.startsWith(path)
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

async function handleNewChat() {
  const session = await chatStore.createSession()
  if (session) {
    chatStore.selectSession(session.id)
    router.push('/')
  }
}

function selectSession(sessionId: string) {
  chatStore.selectSession(sessionId)
  router.push('/')
}

async function handleSessionCommand(command: string, sessionId: string) {
  if (command === 'rename') {
    const session = chatStore.sessions.find(s => s.id === sessionId)
    if (!session) return
    
    const { value } = await ElMessageBox.prompt('请输入新名称', '重命名会话', {
      inputValue: session.title,
      confirmButtonText: '确定',
      cancelButtonText: '取消',
    })
    
    if (value) {
      await chatStore.renameSession(sessionId, value)
    }
  } else if (command === 'delete') {
    await ElMessageBox.confirm('确定要删除这个会话吗？', '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
    
    await chatStore.deleteSession(sessionId)
  }
}

async function handleUserCommand(command: string) {
  if (command === 'settings') {
    showSettings.value = true
  } else if (command === 'theme') {
    settingsStore.theme = theme.value === 'light' ? 'dark' : 'light'
  } else if (command === 'logout') {
    await ElMessageBox.confirm('确定要退出登录吗？', '退出登录', {
      type: 'warning',
      confirmButtonText: '退出',
      cancelButtonText: '取消',
    })
    
    await authStore.logout()
    router.push('/login')
    ElMessage.success('已退出登录')
  }
}

// Load sessions on mount
onMounted(() => {
  chatStore.loadSessions()
})
</script>

<style lang="scss" scoped>
.main-layout {
  display: flex;
  height: 100vh;
  background-color: var(--bg-color);
}

.sidebar {
  width: 280px;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: var(--bg-surface);
  border-right: 1px solid var(--border-color);
  transition: width 0.3s ease;
  
  &.collapsed {
    width: 64px;
    
    .new-chat-btn {
      padding: 8px;
      justify-content: center;
    }
    
    .nav-item {
      justify-content: center;
      padding: 12px;
    }
  }
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border-color);
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  
  .logo-icon {
    width: 32px;
    height: 32px;
  }
  
  .logo-text {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
}

.collapse-btn {
  color: var(--text-secondary);
}

.sidebar-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.new-chat-btn {
  width: 100%;
  justify-content: flex-start;
  gap: 8px;
}

.nav-menu {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 8px;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 0.2s ease;
  
  &:hover {
    background-color: var(--bg-hover);
    color: var(--text-primary);
  }
  
  &.active {
    background-color: var(--primary-light);
    color: var(--primary-color);
  }
  
  .el-icon {
    font-size: 18px;
  }
}

.session-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.session-list-header {
  font-size: 12px;
  color: var(--text-muted);
  padding: 8px 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.session-items {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  
  &:hover {
    background-color: var(--bg-hover);
    
    .session-menu-btn {
      opacity: 1;
    }
  }
  
  &.active {
    background-color: var(--bg-hover);
  }
  
  .el-icon {
    color: var(--text-muted);
    flex-shrink: 0;
  }
  
  .session-title {
    flex: 1;
    font-size: 13px;
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  
  .session-menu-btn {
    opacity: 0;
    transition: opacity 0.2s ease;
  }
}

.sidebar-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-top: 1px solid var(--border-color);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 12px;
  
  .username {
    font-size: 14px;
    color: var(--text-primary);
  }
}

.main-content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>
