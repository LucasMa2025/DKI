<template>
  <div
    class="message-item"
    :class="[
      `message-${message.role}`,
      { compact: compact }
    ]"
  >
    <!-- Avatar -->
    <div class="message-avatar">
      <el-avatar v-if="message.role === 'user'" :size="avatarSize">
        {{ username?.charAt(0).toUpperCase() }}
      </el-avatar>
      <div v-else class="assistant-avatar" :style="{ width: `${avatarSize}px`, height: `${avatarSize}px` }">
        <img src="/logo.svg" alt="DKI" />
      </div>
    </div>
    
    <!-- Content -->
    <div class="message-content">
      <!-- Header -->
      <div class="message-header" v-if="showHeader">
        <span class="message-author">
          {{ message.role === 'user' ? username : 'DKI Assistant' }}
        </span>
        <span class="message-time" v-if="showTimestamp">
          {{ formatTime(message.timestamp) }}
        </span>
      </div>
      
      <!-- Body -->
      <div class="message-body" :class="{ loading: isLoading }">
        <!-- Loading state -->
        <template v-if="isLoading">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </template>
        
        <!-- Content -->
        <template v-else>
          <div
            v-if="message.role === 'assistant'"
            class="markdown-content"
            v-html="renderedContent"
          />
          <div v-else class="plain-content">{{ message.content }}</div>
        </template>
      </div>
      
      <!-- DKI Metadata -->
      <div v-if="message.dkiMetadata && showMetadata" class="dki-metadata">
        <el-tooltip content="缓存层级">
          <el-tag size="small" :type="message.dkiMetadata.cacheHit ? 'success' : 'info'">
            {{ message.dkiMetadata.cacheTier || 'COMPUTE' }}
          </el-tag>
        </el-tooltip>
        <el-tooltip content="注入强度">
          <el-tag size="small">α={{ message.dkiMetadata.alpha?.toFixed(2) }}</el-tag>
        </el-tooltip>
        <el-tooltip content="响应延迟">
          <el-tag size="small">{{ message.dkiMetadata.latencyMs }}ms</el-tag>
        </el-tooltip>
        <el-tooltip v-if="message.dkiMetadata.preferenceTokens" content="偏好 Tokens">
          <el-tag size="small" type="warning">
            偏好: {{ message.dkiMetadata.preferenceTokens }}
          </el-tag>
        </el-tooltip>
        <el-tooltip v-if="message.dkiMetadata.historyTokens" content="历史 Tokens">
          <el-tag size="small" type="info">
            历史: {{ message.dkiMetadata.historyTokens }}
          </el-tag>
        </el-tooltip>
      </div>
      
      <!-- Actions -->
      <div class="message-actions" v-if="showActions && !isLoading">
        <el-button-group size="small">
          <el-tooltip content="复制">
            <el-button :icon="CopyDocument" text @click="handleCopy" />
          </el-tooltip>
          <el-tooltip content="重新生成" v-if="message.role === 'assistant'">
            <el-button :icon="Refresh" text @click="$emit('regenerate')" />
          </el-tooltip>
        </el-button-group>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { CopyDocument, Refresh } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { renderMarkdown } from '@/utils/markdown'
import type { ChatMessage } from '@/types'
import dayjs from 'dayjs'

const props = withDefaults(defineProps<{
  message: ChatMessage
  username?: string
  showTimestamp?: boolean
  showMetadata?: boolean
  showActions?: boolean
  showHeader?: boolean
  compact?: boolean
  avatarSize?: number
}>(), {
  username: 'User',
  showTimestamp: true,
  showMetadata: false,
  showActions: true,
  showHeader: true,
  compact: false,
  avatarSize: 36,
})

defineEmits<{
  (e: 'regenerate'): void
}>()

const isLoading = computed(() => {
  return props.message.content === '' && props.message.role === 'assistant'
})

const renderedContent = computed(() => {
  return renderMarkdown(props.message.content)
})

function formatTime(timestamp: string) {
  return dayjs(timestamp).format('HH:mm')
}

async function handleCopy() {
  try {
    await navigator.clipboard.writeText(props.message.content)
    ElMessage.success('已复制到剪贴板')
  } catch {
    ElMessage.error('复制失败')
  }
}
</script>

<style lang="scss" scoped>
.message-item {
  display: flex;
  gap: 16px;
  padding: 16px 0;
  
  &.compact {
    padding: 8px 0;
    gap: 12px;
    
    .message-body {
      padding: 8px 12px;
    }
  }
  
  &.message-user {
    .message-body {
      background-color: var(--primary-color);
      color: white;
      border-radius: 16px 16px 4px 16px;
    }
  }
  
  &.message-assistant {
    .message-body {
      background-color: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: 16px 16px 16px 4px;
    }
  }
  
  &:hover {
    .message-actions {
      opacity: 1;
    }
  }
}

.message-avatar {
  flex-shrink: 0;
  
  .assistant-avatar {
    border-radius: 50%;
    background: linear-gradient(135deg, #10b981, #059669);
    display: flex;
    align-items: center;
    justify-content: center;
    
    img {
      width: 60%;
      height: 60%;
      filter: brightness(0) invert(1);
    }
  }
}

.message-content {
  flex: 1;
  min-width: 0;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  
  .message-author {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .message-time {
    font-size: 12px;
    color: var(--text-muted);
  }
}

.message-body {
  padding: 12px 16px;
  max-width: 80%;
  
  &.loading {
    padding: 16px 24px;
  }
}

.typing-indicator {
  display: flex;
  gap: 4px;
  
  span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--text-muted);
    animation: typing 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: 0s; }
    &:nth-child(2) { animation-delay: 0.2s; }
    &:nth-child(3) { animation-delay: 0.4s; }
  }
}

@keyframes typing {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-8px); }
}

.markdown-content {
  font-size: 14px;
  line-height: 1.7;
  
  :deep(p) {
    margin: 0 0 12px;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  :deep(pre) {
    margin: 12px 0;
    border-radius: 8px;
    overflow-x: auto;
    
    code {
      font-family: 'Fira Code', 'Consolas', monospace;
      font-size: 13px;
    }
  }
  
  :deep(code:not(pre code)) {
    background-color: var(--bg-hover);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
  }
  
  :deep(ul), :deep(ol) {
    margin: 12px 0;
    padding-left: 24px;
  }
  
  :deep(blockquote) {
    margin: 12px 0;
    padding-left: 16px;
    border-left: 4px solid var(--primary-color);
    color: var(--text-secondary);
  }
}

.plain-content {
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.dki-metadata {
  display: flex;
  gap: 8px;
  margin-top: 8px;
  flex-wrap: wrap;
}

.message-actions {
  margin-top: 8px;
  opacity: 0;
  transition: opacity 0.2s ease;
}
</style>
