<template>
  <div class="preferences-view">
    <!-- Header -->
    <header class="page-header">
      <div class="header-content">
        <h1>Preferences</h1>
        <p>Manage your personalized preferences. DKI will use them to provide customized responses.</p>
      </div>
      <el-button type="primary" :icon="Plus" @click="showCreateDialog = true">
        Add Preference
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
          <span class="stat-label">Total Preferences</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon active">
          <el-icon><CircleCheck /></el-icon>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ activePreferences.length }}</span>
          <span class="stat-label">Enabled</span>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon categories">
          <el-icon><Folder /></el-icon>
        </div>
        <div class="stat-info">
          <span class="stat-value">{{ Object.keys(preferencesByCategory).length }}</span>
          <span class="stat-label">Categories</span>
        </div>
      </div>
    </div>
    
    <!-- Category Tabs -->
    <el-tabs v-model="activeCategory" class="category-tabs">
      <el-tab-pane label="All" name="all" />
      <el-tab-pane
        v-for="(prefs, category) in preferencesByCategory"
        :key="category"
        :label="category"
        :name="category"
      />
    </el-tabs>
    
    <!-- Preferences List -->
    <div class="preferences-list" v-loading="loading">
      <el-empty v-if="filteredPreferences.length === 0" description="No preferences set">
        <el-button type="primary" @click="showCreateDialog = true">Add Preference</el-button>
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
                    <el-icon><Edit /></el-icon>Edit
                  </el-dropdown-item>
                  <el-dropdown-item command="duplicate">
                    <el-icon><CopyDocument /></el-icon>Duplicate
                  </el-dropdown-item>
                  <el-dropdown-item command="delete" divided>
                    <el-icon><Delete /></el-icon>Delete
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
            Priority: {{ pref.priority }}
          </span>
          <span class="pref-date" v-if="pref.updatedAt">
            Updated {{ formatDate(pref.updatedAt) }}
          </span>
        </div>
      </div>
    </div>
    
    <!-- Create/Edit Dialog -->
    <el-dialog
      v-model="showCreateDialog"
      :title="editingPref ? 'Edit Preference' : 'Add Preference'"
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
        <el-form-item label="Preference Type" prop="preferenceType">
          <el-select v-model="form.preferenceType" placeholder="Select type" style="width: 100%">
            <el-option label="General" value="general" />
            <el-option label="Language Style" value="style" />
            <el-option label="Technical" value="technical" />
            <el-option label="Format" value="format" />
            <el-option label="Domain Knowledge" value="domain" />
            <el-option label="Other" value="other" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="Category" prop="category">
          <el-select
            v-model="form.category"
            placeholder="Select or enter category"
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
        
        <el-form-item label="Preference Content" prop="preferenceText">
          <el-input
            v-model="form.preferenceText"
            type="textarea"
            :rows="4"
            placeholder="Describe your preference, e.g.: I prefer concise code style, favor TypeScript"
            maxlength="500"
            show-word-limit
          />
        </el-form-item>
        
        <el-form-item label="Priority" prop="priority">
          <el-slider
            v-model="form.priority"
            :min="0"
            :max="10"
            :step="1"
            show-stops
            :marks="priorityMarks"
          />
        </el-form-item>
        
        <el-form-item label="Status">
          <el-switch v-model="form.isActive" active-text="Enabled" inactive-text="Disabled" />
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showCreateDialog = false">Cancel</el-button>
        <el-button type="primary" :loading="saving" @click="handleSave">
          {{ editingPref ? 'Save' : 'Add' }}
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
    { required: true, message: 'Please select preference type', trigger: 'change' },
  ],
  preferenceText: [
    { required: true, message: 'Please enter preference content', trigger: 'blur' },
    { min: 5, max: 500, message: 'Preference content must be 5-500 characters', trigger: 'blur' },
  ],
}

const priorityMarks = {
  0: 'Low',
  5: 'Med',
  10: 'High',
}

const loading = computed(() => preferencesStore.loading)
const preferences = computed(() => preferencesStore.preferences)
const activePreferences = computed(() => preferencesStore.activePreferences)
const preferencesByCategory = computed(() => preferencesStore.preferencesByCategory)

const existingCategories = computed(() => {
  return Object.keys(preferencesByCategory.value).filter(c => c !== 'Uncategorized')
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
  ElMessage.success(pref.isActive ? 'Preference enabled' : 'Preference disabled')
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
  await ElMessageBox.confirm('Are you sure you want to delete this preference?', 'Delete Preference', {
    type: 'warning',
    confirmButtonText: 'Delete',
    cancelButtonText: 'Cancel',
  })
  
  const success = await preferencesStore.deletePreference(pref.id!)
  if (success) {
    ElMessage.success('Preference deleted')
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
        ElMessage.success('Preference updated')
      } else {
        await preferencesStore.createPreference({
          preferenceType: form.preferenceType,
          category: form.category || undefined,
          preferenceText: form.preferenceText,
          priority: form.priority,
          isActive: form.isActive,
        })
        ElMessage.success('Preference added')
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
