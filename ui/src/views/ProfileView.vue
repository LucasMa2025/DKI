<template>
  <div class="profile-page">
    <div class="page-header">
      <h1>User Profile</h1>
      <p class="subtitle">Manage your account settings and password</p>
    </div>

    <div class="profile-content">
      <!-- Profile Card -->
      <el-card class="profile-card" shadow="never">
        <template #header>
          <div class="card-header">
            <el-icon><User /></el-icon>
            <span>Profile Information</span>
          </div>
        </template>

        <el-form
          ref="profileFormRef"
          :model="profileForm"
          :rules="profileRules"
          label-position="top"
          class="profile-form"
        >
          <div class="avatar-section">
            <el-avatar :size="80" :src="authStore.user?.avatar">
              {{ authStore.user?.username?.charAt(0).toUpperCase() }}
            </el-avatar>
            <div class="avatar-info">
              <h3>{{ authStore.user?.username }}</h3>
              <p v-if="authStore.user?.email">{{ authStore.user?.email }}</p>
              <p v-else class="text-muted">No email set</p>
            </div>
          </div>

          <el-divider />

          <el-form-item label="Display Name" prop="display_name">
            <el-input
              v-model="profileForm.display_name"
              placeholder="Enter display name"
              :prefix-icon="User"
            />
          </el-form-item>

          <el-form-item label="Email" prop="email">
            <el-input
              v-model="profileForm.email"
              placeholder="Enter email address"
              :prefix-icon="Message"
            />
          </el-form-item>

          <el-form-item>
            <el-button
              type="primary"
              :loading="profileLoading"
              @click="handleUpdateProfile"
            >
              Save Changes
            </el-button>
          </el-form-item>
        </el-form>
      </el-card>

      <!-- Change Password Card -->
      <el-card class="password-card" shadow="never">
        <template #header>
          <div class="card-header">
            <el-icon><Lock /></el-icon>
            <span>Change Password</span>
            <el-tag v-if="!authStore.user?.hasPassword" type="info" size="small" class="demo-tag">
              Demo Mode (no password set)
            </el-tag>
          </div>
        </template>

        <el-form
          ref="passwordFormRef"
          :model="passwordForm"
          :rules="passwordRules"
          label-position="top"
          class="password-form"
        >
          <el-form-item
            v-if="authStore.user?.hasPassword"
            label="Current Password"
            prop="old_password"
          >
            <el-input
              v-model="passwordForm.old_password"
              type="password"
              placeholder="Enter current password"
              show-password
              :prefix-icon="Lock"
            />
          </el-form-item>

          <el-alert
            v-else
            type="info"
            :closable="false"
            show-icon
            style="margin-bottom: 20px"
          >
            <template #title>
              You are in demo mode (no password set). Set a password to secure your account.
            </template>
          </el-alert>

          <el-form-item label="New Password" prop="new_password">
            <el-input
              v-model="passwordForm.new_password"
              type="password"
              placeholder="Enter new password (min 4 characters)"
              show-password
              :prefix-icon="Lock"
            />
          </el-form-item>

          <el-form-item label="Confirm New Password" prop="confirm_password">
            <el-input
              v-model="passwordForm.confirm_password"
              type="password"
              placeholder="Re-enter new password"
              show-password
              :prefix-icon="Lock"
            />
          </el-form-item>

          <el-form-item>
            <el-button
              type="warning"
              :loading="passwordLoading"
              @click="handleChangePassword"
            >
              {{ authStore.user?.hasPassword ? 'Change Password' : 'Set Password' }}
            </el-button>
          </el-form-item>
        </el-form>
      </el-card>

      <!-- Recover Password Card -->
      <el-card class="recover-card" shadow="never">
        <template #header>
          <div class="card-header">
            <el-icon><Key /></el-icon>
            <span>Password Recovery</span>
          </div>
        </template>

        <p class="recover-description">
          Forgot your password? Enter the email address associated with your account
          and set a new password.
        </p>

        <el-form
          ref="recoverFormRef"
          :model="recoverForm"
          :rules="recoverRules"
          label-position="top"
          class="recover-form"
        >
          <el-form-item label="Email Address" prop="email">
            <el-input
              v-model="recoverForm.email"
              placeholder="Enter your registered email"
              :prefix-icon="Message"
            />
          </el-form-item>

          <el-form-item label="New Password" prop="new_password">
            <el-input
              v-model="recoverForm.new_password"
              type="password"
              placeholder="Enter new password"
              show-password
              :prefix-icon="Lock"
            />
          </el-form-item>

          <el-form-item label="Confirm Password" prop="confirm_password">
            <el-input
              v-model="recoverForm.confirm_password"
              type="password"
              placeholder="Re-enter new password"
              show-password
              :prefix-icon="Lock"
            />
          </el-form-item>

          <el-form-item>
            <el-button
              type="danger"
              :loading="recoverLoading"
              @click="handleRecoverPassword"
            >
              Reset Password
            </el-button>
          </el-form-item>
        </el-form>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { User, Lock, Message, Key } from '@element-plus/icons-vue'
import { ElMessage, FormInstance, FormRules } from 'element-plus'
import { useAuthStore } from '@/stores/auth'
import { api } from '@/services/api'

const authStore = useAuthStore()

const profileFormRef = ref<FormInstance>()
const passwordFormRef = ref<FormInstance>()
const recoverFormRef = ref<FormInstance>()

const profileLoading = ref(false)
const passwordLoading = ref(false)
const recoverLoading = ref(false)

// Profile form
const profileForm = reactive({
  display_name: authStore.user?.username || '',
  email: authStore.user?.email || '',
})

// Password form
const passwordForm = reactive({
  old_password: '',
  new_password: '',
  confirm_password: '',
})

// Recover form
const recoverForm = reactive({
  email: '',
  new_password: '',
  confirm_password: '',
})

// Validation rules
const profileRules: FormRules = {
  display_name: [
    { max: 128, message: 'Display name must be less than 128 characters', trigger: 'blur' },
  ],
  email: [
    { type: 'email', message: 'Please enter a valid email address', trigger: 'blur' },
  ],
}

const passwordRules: FormRules = {
  old_password: authStore.user?.hasPassword
    ? [{ required: true, message: 'Please enter current password', trigger: 'blur' }]
    : [],
  new_password: [
    { required: true, message: 'Please enter new password', trigger: 'blur' },
    { min: 4, max: 128, message: 'Password must be 4-128 characters', trigger: 'blur' },
  ],
  confirm_password: [
    { required: true, message: 'Please confirm new password', trigger: 'blur' },
    {
      validator: (_rule, value, callback) => {
        if (value !== passwordForm.new_password) {
          callback(new Error('Passwords do not match'))
        } else {
          callback()
        }
      },
      trigger: 'blur',
    },
  ],
}

const recoverRules: FormRules = {
  email: [
    { required: true, message: 'Please enter email address', trigger: 'blur' },
    { type: 'email', message: 'Please enter a valid email address', trigger: 'blur' },
  ],
  new_password: [
    { required: true, message: 'Please enter new password', trigger: 'blur' },
    { min: 4, max: 128, message: 'Password must be 4-128 characters', trigger: 'blur' },
  ],
  confirm_password: [
    { required: true, message: 'Please confirm new password', trigger: 'blur' },
    {
      validator: (_rule, value, callback) => {
        if (value !== recoverForm.new_password) {
          callback(new Error('Passwords do not match'))
        } else {
          callback()
        }
      },
      trigger: 'blur',
    },
  ],
}

// Handlers
async function handleUpdateProfile() {
  if (!profileFormRef.value) return

  await profileFormRef.value.validate(async (valid) => {
    if (!valid) return

    profileLoading.value = true
    try {
      const updates: Record<string, string> = {}
      if (profileForm.display_name) updates.display_name = profileForm.display_name
      if (profileForm.email) updates.email = profileForm.email

      const updatedUser = await api.auth.updateProfile(updates)
      
      // Update local store
      if (authStore.user) {
        authStore.user = {
          ...authStore.user,
          ...updatedUser,
        }
      }

      ElMessage.success('Profile updated successfully')
    } catch (error) {
      ElMessage.error(error instanceof Error ? error.message : 'Failed to update profile')
    } finally {
      profileLoading.value = false
    }
  })
}

async function handleChangePassword() {
  if (!passwordFormRef.value) return

  await passwordFormRef.value.validate(async (valid) => {
    if (!valid) return

    passwordLoading.value = true
    try {
      await api.auth.changePassword({
        old_password: passwordForm.old_password,
        new_password: passwordForm.new_password,
      })

      // Update hasPassword flag
      if (authStore.user) {
        authStore.user = {
          ...authStore.user,
          hasPassword: true,
        }
      }

      // Reset form
      passwordForm.old_password = ''
      passwordForm.new_password = ''
      passwordForm.confirm_password = ''

      ElMessage.success('Password changed successfully')
    } catch (error) {
      ElMessage.error(error instanceof Error ? error.message : 'Failed to change password')
    } finally {
      passwordLoading.value = false
    }
  })
}

async function handleRecoverPassword() {
  if (!recoverFormRef.value) return

  await recoverFormRef.value.validate(async (valid) => {
    if (!valid) return

    recoverLoading.value = true
    try {
      const result = await api.auth.recoverPassword({
        email: recoverForm.email,
        new_password: recoverForm.new_password,
      })

      // Reset form
      recoverForm.email = ''
      recoverForm.new_password = ''
      recoverForm.confirm_password = ''

      ElMessage.success(result.message || 'Password has been reset successfully')
    } catch (error) {
      ElMessage.error(error instanceof Error ? error.message : 'Failed to recover password')
    } finally {
      recoverLoading.value = false
    }
  })
}

// Initialize form with current user data
onMounted(async () => {
  try {
    await authStore.refreshUser()
    if (authStore.user) {
      profileForm.display_name = authStore.user.username || ''
      profileForm.email = authStore.user.email || ''
    }
  } catch {
    // Ignore refresh errors
  }
})
</script>

<style lang="scss" scoped>
.profile-page {
  padding: 24px;
  max-width: 800px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 32px;

  h1 {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 8px;
  }

  .subtitle {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0;
  }
}

.profile-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.profile-card,
.password-card,
.recover-card {
  border-radius: 16px;
  border: 1px solid var(--border-color);

  :deep(.el-card__header) {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color);
  }

  :deep(.el-card__body) {
    padding: 24px;
  }
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);

  .el-icon {
    font-size: 20px;
  }

  .demo-tag {
    margin-left: auto;
  }
}

.avatar-section {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 8px;

  h3 {
    margin: 0 0 4px;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  p {
    margin: 0;
    font-size: 14px;
    color: var(--text-secondary);

    &.text-muted {
      font-style: italic;
      opacity: 0.6;
    }
  }
}

.profile-form,
.password-form,
.recover-form {
  max-width: 480px;

  .el-input {
    --el-input-border-radius: 10px;
  }
}

.recover-description {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0 0 20px;
  line-height: 1.6;
}

@media (max-width: 768px) {
  .profile-page {
    padding: 16px;
  }

  .avatar-section {
    flex-direction: column;
    text-align: center;
  }
}
</style>
