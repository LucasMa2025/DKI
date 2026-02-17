<template>
  <div class="sessions-view">
    <!-- Header -->
    <header class="page-header">
      <div class="header-content">
        <h1>Session Management</h1>
        <p>View and manage all your conversation records</p>
      </div>
      <div class="header-actions">
        <el-input
          v-model="searchQuery"
          placeholder="Search sessions..."
          :prefix-icon="Search"
          clearable
          style="width: 240px"
        />
        <el-button type="primary" :icon="Plus" @click="handleNewSession">
          New Session
        </el-button>
      </div>
    </header>
    
    <!-- Stats -->
    <div class="stats-row">
      <div class="stat-item">
        <span class="stat-value">{{ sessions.length }}</span>
        <span class="stat-label">Total Sessions</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">{{ totalMessages }}</span>
        <span class="stat-label">Total Messages</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">{{ todaySessions }}</span>
        <span class="stat-label">Today's Sessions</span>
      </div>
    </div>
    
    <!-- Sessions Table -->
    <div class="sessions-table-container">
      <el-table
        :data="filteredSessions"
        style="width: 100%"
        :row-class-name="getRowClassName"
        @row-click="handleRowClick"
        v-loading="loading"
      >
        <el-table-column type="selection" width="50" />
        
        <el-table-column label="Session Title" min-width="200">
          <template #default="{ row }">
            <div class="session-title-cell">
              <el-icon class="session-icon"><ChatDotRound /></el-icon>
              <span class="session-title">{{ row.title }}</span>
              <el-tag
                v-if="row.id === currentSessionId"
                type="success"
                size="small"
              >
                Current
              </el-tag>
            </div>
          </template>
        </el-table-column>
        
        <el-table-column label="Messages" width="100" align="center">
          <template #default="{ row }">
            <span class="message-count">{{ row.messageCount }}</span>
          </template>
        </el-table-column>
        
        <el-table-column label="Preview" min-width="200">
          <template #default="{ row }">
            <span class="session-preview">{{ row.preview || 'No content yet' }}</span>
          </template>
        </el-table-column>
        
        <el-table-column label="Created" width="160">
          <template #default="{ row }">
            {{ formatDate(row.createdAt) }}
          </template>
        </el-table-column>
        
        <el-table-column label="Updated" width="160">
          <template #default="{ row }">
            {{ formatDate(row.updatedAt) }}
          </template>
        </el-table-column>
        
        <el-table-column label="Actions" width="150" fixed="right">
          <template #default="{ row }">
            <el-button-group>
              <el-tooltip content="Open">
                <el-button :icon="View" text @click.stop="handleOpen(row)" />
              </el-tooltip>
              <el-tooltip content="Rename">
                <el-button :icon="Edit" text @click.stop="handleRename(row)" />
              </el-tooltip>
              <el-tooltip content="Export">
                <el-button :icon="Download" text @click.stop="handleExport(row)" />
              </el-tooltip>
              <el-tooltip content="Delete">
                <el-button :icon="Delete" text type="danger" @click.stop="handleDelete(row)" />
              </el-tooltip>
            </el-button-group>
          </template>
        </el-table-column>
      </el-table>
    </div>
    
    <!-- Pagination -->
    <div class="pagination-container">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50, 100]"
        :total="filteredSessions.length"
        layout="total, sizes, prev, pager, next, jumper"
        background
      />
    </div>
    
    <!-- Session Detail Drawer -->
    <el-drawer
      v-model="showDetail"
      :title="selectedSession?.title || 'Session Details'"
      direction="rtl"
      size="500px"
    >
      <div v-if="selectedSession">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="Session ID">
            {{ selectedSession.id }}
          </el-descriptions-item>
          <el-descriptions-item label="Message Count">
            {{ selectedSession.messageCount }}
          </el-descriptions-item>
          <el-descriptions-item label="Created">
            {{ formatDate(selectedSession.createdAt) }}
          </el-descriptions-item>
          <el-descriptions-item label="Updated">
            {{ formatDate(selectedSession.updatedAt) }}
          </el-descriptions-item>
        </el-descriptions>
        
        <h4>Message Preview</h4>
        <div class="messages-preview" v-loading="loadingMessages">
          <div
            v-for="msg in previewMessages"
            :key="msg.id"
            class="preview-message"
            :class="[`message-${msg.role}`]"
          >
            <div class="message-role">{{ msg.role === 'user' ? 'User' : 'Assistant' }}</div>
            <div class="message-content">{{ msg.content }}</div>
            <div class="message-time">{{ formatTime(msg.timestamp) }}</div>
          </div>
          <el-empty v-if="previewMessages.length === 0" description="No messages" />
        </div>
      </div>
      <el-empty v-else description="Please select a session" />
      
      <template #footer>
        <el-button @click="showDetail = false">Close</el-button>
        <el-button type="primary" @click="handleOpenFromDetail" :disabled="!selectedSession">
          Open Session
        </el-button>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import {
  Search,
  Plus,
  ChatDotRound,
  View,
  Edit,
  Download,
  Delete,
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useChatStore } from '@/stores/chat'
import type { Session, ChatMessage } from '@/types'
import { api } from '@/services/api'
import dayjs from 'dayjs'

const router = useRouter()
const chatStore = useChatStore()

const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(20)
const showDetail = ref(false)
const selectedSession = ref<Session | null>(null)
const previewMessages = ref<ChatMessage[]>([])
const loadingMessages = ref(false)

const loading = computed(() => false) // Would be from store
const sessions = computed(() => chatStore.sessions)
const currentSessionId = computed(() => chatStore.currentSessionId)

const filteredSessions = computed(() => {
  if (!searchQuery.value) return sessions.value
  const query = searchQuery.value.toLowerCase()
  return sessions.value.filter(s =>
    s.title.toLowerCase().includes(query) ||
    s.preview?.toLowerCase().includes(query)
  )
})

const totalMessages = computed(() => {
  return sessions.value.reduce((sum, s) => sum + s.messageCount, 0)
})

const todaySessions = computed(() => {
  const today = dayjs().startOf('day')
  return sessions.value.filter(s => dayjs(s.createdAt).isAfter(today)).length
})

function formatDate(date: string) {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

function formatTime(date: string) {
  return dayjs(date).format('HH:mm:ss')
}

function getRowClassName({ row }: { row: Session }) {
  return row.id === currentSessionId.value ? 'current-session-row' : ''
}

async function handleNewSession() {
  const session = await chatStore.createSession()
  if (session) {
    chatStore.selectSession(session.id)
    router.push('/')
  }
}

function handleRowClick(row: Session) {
  selectedSession.value = row
  showDetail.value = true
  loadSessionMessages(row.id)
}

async function loadSessionMessages(sessionId: string) {
  loadingMessages.value = true
  try {
    previewMessages.value = await api.sessions.getMessages(sessionId)
  } catch {
    previewMessages.value = []
  } finally {
    loadingMessages.value = false
  }
}

function handleOpen(session: Session) {
  chatStore.selectSession(session.id)
  router.push('/')
}

function handleOpenFromDetail() {
  if (selectedSession.value) {
    handleOpen(selectedSession.value)
  }
}

async function handleRename(session: Session) {
  const { value } = await ElMessageBox.prompt('Enter new name', 'Rename Session', {
    inputValue: session.title,
    confirmButtonText: 'OK',
    cancelButtonText: 'Cancel',
  })
  
  if (value) {
    await chatStore.renameSession(session.id, value)
    ElMessage.success('Renamed successfully')
  }
}

function handleExport(session: Session) {
  // Export session as JSON
  const data = {
    session,
    exportedAt: new Date().toISOString(),
  }
  
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `session-${session.id}.json`
  a.click()
  URL.revokeObjectURL(url)
  
  ElMessage.success('Export successful')
}

async function handleDelete(session: Session) {
  await ElMessageBox.confirm(
    `Are you sure you want to delete session "${session.title}"? This action cannot be undone.`,
    'Delete Session',
    {
      type: 'warning',
      confirmButtonText: 'Delete',
      cancelButtonText: 'Cancel',
    }
  )
  
  await chatStore.deleteSession(session.id)
  ElMessage.success('Session deleted')
}

onMounted(() => {
  chatStore.loadSessions()
})
</script>

<style lang="scss" scoped>
.sessions-view {
  height: 100%;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  
  .header-content {
    h1 {
      font-size: 24px;
      font-weight: 700;
      color: var(--text-primary);
      margin: 0 0 8px;
    }
    
    p {
      font-size: 14px;
      color: var(--text-secondary);
      margin: 0;
    }
  }
  
  .header-actions {
    display: flex;
    gap: 12px;
  }
}

.stats-row {
  display: flex;
  gap: 32px;
  margin-bottom: 24px;
  padding: 20px;
  background-color: var(--bg-surface);
  border-radius: 12px;
  border: 1px solid var(--border-color);
}

.stat-item {
  display: flex;
  flex-direction: column;
  
  .stat-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
  }
  
  .stat-label {
    font-size: 13px;
    color: var(--text-secondary);
  }
}

.sessions-table-container {
  flex: 1;
  background-color: var(--bg-surface);
  border-radius: 12px;
  border: 1px solid var(--border-color);
  overflow: hidden;
  
  :deep(.el-table) {
    --el-table-bg-color: transparent;
    --el-table-tr-bg-color: transparent;
    --el-table-header-bg-color: var(--bg-hover);
    
    .current-session-row {
      background-color: rgba(16, 185, 129, 0.05);
    }
  }
}

.session-title-cell {
  display: flex;
  align-items: center;
  gap: 8px;
  
  .session-icon {
    color: var(--text-muted);
  }
  
  .session-title {
    font-weight: 500;
    color: var(--text-primary);
  }
}

.message-count {
  font-weight: 600;
  color: var(--primary-color);
}

.session-preview {
  color: var(--text-secondary);
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.messages-preview {
  max-height: 400px;
  overflow-y: auto;
  margin-top: 16px;
}

.preview-message {
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 8px;
  
  &.message-user {
    background-color: var(--bg-hover);
  }
  
  &.message-assistant {
    background-color: rgba(16, 185, 129, 0.05);
    border: 1px solid rgba(16, 185, 129, 0.2);
  }
  
  .message-role {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }
  
  .message-content {
    font-size: 14px;
    color: var(--text-primary);
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  
  .message-time {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 8px;
  }
}

h4 {
  margin: 24px 0 12px;
  color: var(--text-primary);
}
</style>
