<template>
  <div class="preferences-view">
    <!-- Header -->
    <header class="page-header">
      <div class="header-content">
        <h1>偏好设置</h1>
        <p>管理您的个性化偏好，DKI 将根据这些偏好为您提供定制化的回答</p>
      </div>
      <el-button type="primary" :icon="Plus" @click="showCreateDialog = true">
        添加偏好
      </el-button>
    </header>
    
    <!-- Stats Cards -->
    <div class="stats-cards">
      <div class="stat-card">
        <div class="stat-icon">
          <el-icon><Document /></el-icon>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ preferences.length }}</span>
          <span class="stat-label">总偏好数</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon active">
          <el-icon><CircleCheck /></el-icon>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ activePreferences.length }}</span>
          <span class="stat-label">已启用</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon categories">
          <el-icon><Folder /></el-icon>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ Object.keys(preferencesByCategory).length }}</span>
          <span class="stat-label">分类数</span>
        </div>
      </div>
    </div>
    
    <!-- Category Tabs -->
    <el-tabs v-model="activeCategory" class="category-tabs">
      <el-tab-pane label="全部" name="all" />
      <el-tab-pane
        v-for="(prefs, category) in preferencesByCategory"
        :key="category"
        :label="category"
        :name="category"
      />
    </el-tabs>
    
    <!-- Preferences List -->
    <div class="preferences-list" v-loading="loading">
      <el-empty v-if="filteredPreferences.length === 0" description="暂无偏好设置">
        <el-button type="primary" @click="showCreateDialog = true">添加偏好</el-button>
      </el-empty>
      
      <div
        v-for="pref in filteredPreferences"
        :key="pref.id"
        class="preference-card"
        :class="{ inactive: !pref.isActive }"
      >
        <div class="pref-header">
          <div class="pref-type">
            <el-tag :type="getTypeTagType(pref.preferenceType)" size="small">
              {{ pref.preferenceType }}
            </el-tag>
            <el-tag v-if="pref.category" type="info" size="small">
              {{ pref.category }}
            </el-tag>
          </div>
          <div class="pref-actions">
            <el-switch
              v-model="pref.isActive"
              @change="handleToggle(pref)"
              size="small"
            />
            <el-dropdown trigger="click" @command="(cmd: string) => handleCommand(cmd, pref)">
              <el-button :icon="MoreFilled" text size="small" />
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="edit">
                    <el-icon><Edit /></el-icon>编辑
                  </el-dropdown-item>
                  <el-dropdown-item command="duplicate">
                    <el-icon><CopyDocument /></el-icon>复制
                  </el-dropdown-item>
                  <el-dropdown-item command="delete" divided>
                    <el-icon><Delete /></el-icon>删除
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
        
        <div class="pref-content">
          <p class="pref-text">{{ pref.preferenceText }}</p>
        </div>
        
        <div class="pref-footer">
          <span class="pref-priority">
            优先级: {{ pref.priority }}
          </span>
          <span class="pref-date" v-if="pref.updatedAt">
            更新于 {{ formatDate(pref.updatedAt) }}
          </span>
        </div>
      </div>
    </div>
    
    <!-- Create/Edit Dialog -->
    <el-dialog
      v-model="showCreateDialog"
      :title="editingPref ? '编辑偏好' : '添加偏好'"
      width="600px"
      :close-on-click-modal="false"
      @close="resetForm"
    >
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-position="top"
      >
        <el-form-item label="偏好类型" prop="preferenceType">
          <el-select v-model="form.preferenceType" placeholder="选择类型" style="width: 100%">
            <el-option label="通用偏好" value="general" />
            <el-option label="语言风格" value="style" />
            <el-option label="技术偏好" value="technical" />
            <el-option label="格式偏好" value="format" />
            <el-option label="领域知识" value="domain" />
            <el-option label="其他" value="other" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="分类" prop="category">
          <el-select
            v-model="form.category"
            placeholder="选择或输入分类"
            filterable
            allow-create
            style="width: 100%"
          >
            <el-option
              v-for="cat in existingCategories"
              :key="cat"
              :label="cat"
              :value="cat"
            />
          </el-select>
        </el-form-item>
        
        <el-form-item label="偏好内容" prop="preferenceText">
          <el-input
            v-model="form.preferenceText"
            type="textarea"
            :rows="4"
            placeholder="描述您的偏好，例如：我喜欢简洁的代码风格，偏好使用 TypeScript"
            maxlength="500"
            show-word-limit
          />
        </el-form-item>
        
        <el-form-item label="优先级" prop="priority">
          <el-slider
            v-model="form.priority"
            :min="0"
            :max="10"
            :step="1"
            show-stops
            :marks="priorityMarks"
          />
        </el-form-item>
        
        <el-form-item label="状态">
          <el-switch v-model="form.isActive" active-text="启用" inactive-text="禁用" />
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="handleSave">
          {{ editingPref ? '保存' : '添加' }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive, onMounted } from 'vue'
import {
  Plus,
  Document,
  CircleCheck,
  Folder,
  MoreFilled,
  Edit,
  Delete,
  CopyDocument,
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox, FormInstance, FormRules } from 'element-plus'
import { usePreferencesStore } from '@/stores/preferences'
import type { UserPreference } from '@/types'
import dayjs from 'dayjs'

const preferencesStore = usePreferencesStore()

const formRef = ref<FormInstance>()
const showCreateDialog = ref(false)
const editingPref = ref<UserPreference | null>(null)
const saving = ref(false)
const activeCategory = ref('all')

const form = reactive({
  preferenceType: 'general',
  category: '',
  preferenceText: '',
  priority: 5,
  isActive: true,
})

const rules: FormRules = {
  preferenceType: [
    { required: true, message: '请选择偏好类型', trigger: 'change' },
  ],
  preferenceText: [
    { required: true, message: '请输入偏好内容', trigger: 'blur' },
    { min: 5, max: 500, message: '偏好内容长度为 5-500 个字符', trigger: 'blur' },
  ],
}

const priorityMarks = {
  0: '低',
  5: '中',
  10: '高',
}

const loading = computed(() => preferencesStore.loading)
const preferences = computed(() => preferencesStore.preferences)
const activePreferences = computed(() => preferencesStore.activePreferences)
const preferencesByCategory = computed(() => preferencesStore.preferencesByCategory)

const existingCategories = computed(() => {
  return Object.keys(preferencesByCategory.value).filter(c => c !== '未分类')
})

const filteredPreferences = computed(() => {
  if (activeCategory.value === 'all') {
    return preferences.value
  }
  return preferencesByCategory.value[activeCategory.value] || []
})

function getTypeTagType(type: string) {
  const types: Record<string, string> = {
    general: '',
    style: 'success',
    technical: 'warning',
    format: 'info',
    domain: 'danger',
    other: 'info',
  }
  return types[type] || ''
}

function formatDate(date: string) {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

function resetForm() {
  editingPref.value = null
  form.preferenceType = 'general'
  form.category = ''
  form.preferenceText = ''
  form.priority = 5
  form.isActive = true
  formRef.value?.resetFields()
}

async function handleToggle(pref: UserPreference) {
  await preferencesStore.updatePreference(pref.id!, { isActive: pref.isActive })
  ElMessage.success(pref.isActive ? '偏好已启用' : '偏好已禁用')
}

function handleCommand(command: string, pref: UserPreference) {
  if (command === 'edit') {
    editingPref.value = pref
    form.preferenceType = pref.preferenceType
    form.category = pref.category || ''
    form.preferenceText = pref.preferenceText
    form.priority = pref.priority
    form.isActive = pref.isActive
    showCreateDialog.value = true
  } else if (command === 'duplicate') {
    form.preferenceType = pref.preferenceType
    form.category = pref.category || ''
    form.preferenceText = pref.preferenceText
    form.priority = pref.priority
    form.isActive = true
    showCreateDialog.value = true
  } else if (command === 'delete') {
    handleDelete(pref)
  }
}

async function handleDelete(pref: UserPreference) {
  await ElMessageBox.confirm('确定要删除这个偏好吗？', '删除偏好', {
    type: 'warning',
    confirmButtonText: '删除',
    cancelButtonText: '取消',
  })
  
  const success = await preferencesStore.deletePreference(pref.id!)
  if (success) {
    ElMessage.success('偏好已删除')
  }
}

async function handleSave() {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    saving.value = true
    
    try {
      if (editingPref.value) {
        await preferencesStore.updatePreference(editingPref.value.id!, {
          preferenceType: form.preferenceType,
          category: form.category || undefined,
          preferenceText: form.preferenceText,
          priority: form.priority,
          isActive: form.isActive,
        })
        ElMessage.success('偏好已更新')
      } else {
        await preferencesStore.createPreference({
          preferenceType: form.preferenceType,
          category: form.category || undefined,
          preferenceText: form.preferenceText,
          priority: form.priority,
          isActive: form.isActive,
        })
        ElMessage.success('偏好已添加')
      }
      
      showCreateDialog.value = false
      resetForm()
    } finally {
      saving.value = false
    }
  })
}

onMounted(() => {
  preferencesStore.loadPreferences()
})
</script>

<style lang="scss" scoped>
.preferences-view {
  height: 100%;
  overflow-y: auto;
  padding: 24px;
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
}

.stats-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background-color: var(--bg-surface);
  border-radius: 12px;
  border: 1px solid var(--border-color);
  
  .stat-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--bg-hover);
    color: var(--text-secondary);
    
    .el-icon {
      font-size: 24px;
    }
    
    &.active {
      background-color: rgba(16, 185, 129, 0.1);
      color: #10b981;
    }
    
    &.categories {
      background-color: rgba(59, 130, 246, 0.1);
      color: #3b82f6;
    }
  }
  
  .stat-info {
    display: flex;
    flex-direction: column;
    
    .stat-value {
      font-size: 24px;
      font-weight: 700;
      color: var(--text-primary);
    }
    
    .stat-label {
      font-size: 13px;
      color: var(--text-secondary);
    }
  }
}

.category-tabs {
  margin-bottom: 24px;
  
  :deep(.el-tabs__header) {
    margin-bottom: 0;
  }
}

.preferences-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 16px;
}

.preference-card {
  background-color: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 20px;
  transition: all 0.2s ease;
  
  &:hover {
    border-color: var(--primary-color);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  }
  
  &.inactive {
    opacity: 0.6;
  }
}

.pref-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  
  .pref-type {
    display: flex;
    gap: 8px;
  }
  
  .pref-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
}

.pref-content {
  .pref-text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
    margin: 0;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
}

.pref-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
  font-size: 12px;
  color: var(--text-muted);
}

:deep(.el-slider__marks-text) {
  font-size: 12px;
}
</style>
