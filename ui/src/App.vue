<template>
  <el-config-provider :locale="locale">
    <router-view />
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import en from 'element-plus/dist/locale/en.mjs'
import { useSettingsStore } from '@/stores/settings'

const settingsStore = useSettingsStore()

const locale = computed(() => {
  return settingsStore.language === 'zh-CN' ? zhCn : en
})

// Apply theme
watchEffect(() => {
  if (settingsStore.theme === 'dark') {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
})
</script>

<style lang="scss">
html, body, #app {
  height: 100%;
  margin: 0;
  padding: 0;
}
</style>
