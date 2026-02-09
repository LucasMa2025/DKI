<template>
  <div class="chat-input">
    <div class="input-container">
      <!-- Attachment button (optional) -->
      <el-tooltip content="添加附件" v-if="showAttachment">
        <el-button :icon="Paperclip" text class="attachment-btn" />
      </el-tooltip>
      
      <!-- Text input -->
      <el-input
        ref="inputRef"
        v-model="message"
        type="textarea"
        :rows="1"
        :autosize="{ minRows: 1, maxRows: maxRows }"
        :placeholder="placeholder"
        resize="none"
        @keydown="handleKeydown"
        @focus="isFocused = true"
        @blur="isFocused = false"
      />
      
      <!-- Send button -->
      <el-button
        type="primary"
        :icon="Promotion"
        :loading="loading"
        :disabled="!canSend"
        class="send-btn"
        @click="handleSend"
      />
    </div>
    
    <!-- Footer hints -->
    <div class="input-footer" v-if="showFooter">
      <span class="input-hint">
        <template v-if="sendOnEnter">
          Enter 发送，Shift+Enter 换行
        </template>
        <template v-else>
          Ctrl+Enter 发送
        </template>
      </span>
      <span class="char-count" v-if="showCharCount">
        {{ message.length }} / {{ maxLength }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { Paperclip, Promotion } from '@element-plus/icons-vue'

const props = withDefaults(defineProps<{
  modelValue?: string
  placeholder?: string
  loading?: boolean
  disabled?: boolean
  sendOnEnter?: boolean
  showAttachment?: boolean
  showFooter?: boolean
  showCharCount?: boolean
  maxRows?: number
  maxLength?: number
}>(), {
  modelValue: '',
  placeholder: '输入消息...',
  loading: false,
  disabled: false,
  sendOnEnter: true,
  showAttachment: false,
  showFooter: true,
  showCharCount: false,
  maxRows: 6,
  maxLength: 4000,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'send', message: string): void
}>()

const inputRef = ref()
const isFocused = ref(false)

const message = computed({
  get: () => props.modelValue,
  set: (value) => emit('update:modelValue', value),
})

const canSend = computed(() => {
  return message.value.trim().length > 0 && !props.loading && !props.disabled
})

function handleKeydown(e: KeyboardEvent) {
  if (props.sendOnEnter) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  } else {
    if (e.key === 'Enter' && e.ctrlKey) {
      e.preventDefault()
      handleSend()
    }
  }
}

function handleSend() {
  if (!canSend.value) return
  
  const text = message.value.trim()
  emit('send', text)
  message.value = ''
  
  // Focus back to input
  inputRef.value?.focus()
}

function focus() {
  inputRef.value?.focus()
}

defineExpose({
  focus,
})
</script>

<style lang="scss" scoped>
.chat-input {
  padding: 16px;
  background-color: var(--bg-surface);
  border-top: 1px solid var(--border-color);
}

.input-container {
  display: flex;
  gap: 12px;
  align-items: flex-end;
  
  .attachment-btn {
    flex-shrink: 0;
    color: var(--text-secondary);
    
    &:hover {
      color: var(--primary-color);
    }
  }
  
  .el-input {
    flex: 1;
    
    :deep(.el-textarea__inner) {
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 14px;
      resize: none;
      background-color: var(--bg-color);
      border-color: var(--border-color);
      
      &:focus {
        border-color: var(--primary-color);
      }
    }
  }
  
  .send-btn {
    flex-shrink: 0;
    height: 44px;
    width: 44px;
    border-radius: 12px;
  }
}

.input-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  padding: 0 4px;
  
  .input-hint {
    font-size: 12px;
    color: var(--text-muted);
  }
  
  .char-count {
    font-size: 12px;
    color: var(--text-muted);
  }
}
</style>
