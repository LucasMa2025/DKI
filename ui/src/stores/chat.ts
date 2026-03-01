import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { ChatMessage, Session, ChatRequest } from "@/types";
import { api } from "@/services/api";
import { useAuthStore } from "./auth";
import { useSettingsStore } from "./settings";

export const useChatStore = defineStore("chat", () => {
    const sessions = ref<Session[]>([]);
    const currentSessionId = ref<string | null>(null);
    const messages = ref<ChatMessage[]>([]);
    const loading = ref(false);
    const streaming = ref(false);
    const error = ref<string | null>(null);

    const currentSession = computed(() => {
        return (
            sessions.value.find((s) => s.id === currentSessionId.value) || null
        );
    });

    // Load sessions
    async function loadSessions() {
        try {
            sessions.value = await api.sessions.list();
        } catch (e) {
            console.error("Failed to load sessions:", e);
        }
    }

    // Create new session
    async function createSession(title?: string): Promise<Session | null> {
        try {
            const session = await api.sessions.create(title || "New Chat");
            sessions.value.unshift(session);
            return session;
        } catch (e) {
            error.value = "Failed to create session";
            return null;
        }
    }

    // Select session
    async function selectSession(sessionId: string) {
        if (currentSessionId.value === sessionId) return;

        currentSessionId.value = sessionId;
        messages.value = [];

        try {
            loading.value = true;
            messages.value = await api.sessions.getMessages(sessionId);
        } catch (e) {
            error.value = "Failed to load messages";
        } finally {
            loading.value = false;
        }
    }

    // Delete session
    async function deleteSession(sessionId: string) {
        try {
            await api.sessions.delete(sessionId);
            sessions.value = sessions.value.filter((s) => s.id !== sessionId);

            if (currentSessionId.value === sessionId) {
                currentSessionId.value = sessions.value[0]?.id || null;
                messages.value = [];
            }
        } catch (e) {
            error.value = "Failed to delete session";
        }
    }

    // Rename session
    async function renameSession(sessionId: string, title: string) {
        try {
            await api.sessions.update(sessionId, { title });
            const session = sessions.value.find((s) => s.id === sessionId);
            if (session) {
                session.title = title;
            }
        } catch (e) {
            error.value = "Failed to rename session";
        }
    }

    /**
     * 从用户消息内容生成可读的会话标题
     * 规则:
     * 1. 去除换行和多余空白
     * 2. 如果内容是问句, 保留问号
     * 3. 截取前 30 个字符 (中文友好)
     * 4. 如果被截断, 添加 "..."
     */
    function generateSessionTitle(content: string): string {
        const cleaned = content.trim().replace(/\s+/g, ' ');
        const maxLen = 30;
        if (cleaned.length <= maxLen) {
            return cleaned;
        }
        // 尝试在标点或空格处截断, 避免截断到词中间
        const truncated = cleaned.slice(0, maxLen);
        const lastPunct = Math.max(
            truncated.lastIndexOf('，'),
            truncated.lastIndexOf(','),
            truncated.lastIndexOf('。'),
            truncated.lastIndexOf(' '),
            truncated.lastIndexOf('、'),
            truncated.lastIndexOf('？'),
            truncated.lastIndexOf('?'),
        );
        if (lastPunct > maxLen * 0.5) {
            return truncated.slice(0, lastPunct) + '...';
        }
        return truncated + '...';
    }

    // Send message
    // 修正: 只传递 user_id 和原始输入，移除 prompt 拼接逻辑
    // DKI 会自动通过适配器读取用户偏好和历史消息进行注入
    async function sendMessage(content: string) {
        const authStore = useAuthStore();
        const settingsStore = useSettingsStore();

        if (!content.trim()) return;

        // 记录是否是新会话 (用于自动命名)
        const isNewSession = !currentSessionId.value;

        // Ensure we have a session
        if (!currentSessionId.value) {
            const session = await createSession();
            if (!session) return;
            currentSessionId.value = session.id;
        }

        // Add user message to local display
        const userMessage: ChatMessage = {
            id: `temp-${Date.now()}`,
            sessionId: currentSessionId.value,
            role: "user",
            content: content.trim(),
            timestamp: new Date().toISOString(),
        };
        messages.value.push(userMessage);

        // 修正: 简化请求，只传递必要信息
        // - user_id: 用户标识 (DKI 用于读取偏好和历史)
        // - session_id: 会话标识 (DKI 用于读取会话历史)
        // - query: 原始用户输入 (不含任何 prompt 构造)
        // DKI 会自动:
        // 1. 通过适配器读取用户偏好 → K/V 注入
        // 2. 通过适配器检索相关历史 → 后缀提示词
        const request: ChatRequest = {
            // 原始用户输入，不拼接任何历史消息或 prompt
            query: content.trim(),
            // 用户标识 - DKI 用于读取偏好和历史
            dkiUserId: authStore.user?.id,
            // 会话标识 - DKI 用于读取会话历史
            dkiSessionId: currentSessionId.value,
            // 可选参数
            model: settingsStore.defaultModel,
            temperature: settingsStore.temperature,
            maxTokens: settingsStore.maxTokens,
            stream: false,
        };

        // Add placeholder for assistant message
        const assistantMessage: ChatMessage = {
            id: `temp-assistant-${Date.now()}`,
            sessionId: currentSessionId.value,
            role: "assistant",
            content: "",
            timestamp: new Date().toISOString(),
        };
        messages.value.push(assistantMessage);

        try {
            loading.value = true;
            error.value = null;

            const response = await api.chat.send(request);

            // Update assistant message
            const lastMessage = messages.value[messages.value.length - 1];
            if (lastMessage.role === "assistant") {
                lastMessage.id = response.id;
                // 优先从 choices 获取，fallback 到 text 字段 (兼容 DKIChatResponse)
                lastMessage.content =
                    response.choices?.[0]?.message?.content ||
                    (response as any).text ||
                    "";
                lastMessage.dkiMetadata = response.dkiMetadata;
            }

            // Update session preview
            const session = sessions.value.find(
                (s) => s.id === currentSessionId.value
            );
            if (session) {
                session.preview = content.slice(0, 50);
                session.messageCount = messages.value.length;
                session.updatedAt = new Date().toISOString();
            }

            // 自动命名: 新会话的第一条消息发送成功后, 用消息内容生成可读标题
            if (isNewSession && currentSessionId.value) {
                const autoTitle = generateSessionTitle(content);
                // 异步更新标题, 不阻塞主流程
                renameSession(currentSessionId.value, autoTitle).catch((e) => {
                    console.warn('Auto-rename session failed:', e);
                });
            }
        } catch (e) {
            const errMsg =
                e instanceof Error ? e.message : "Failed to send message";
            error.value = errMsg;

            // v5.8: 超时或网络错误时，在助手消息中显示错误提示，而非静默删除
            // 这样用户能清楚看到请求失败的原因 (如推理超时)
            const lastMessage = messages.value[messages.value.length - 1];
            if (lastMessage?.role === "assistant" && !lastMessage.content) {
                const isTimeout =
                    errMsg.toLowerCase().includes("timeout") ||
                    errMsg.includes("ECONNABORTED") ||
                    errMsg.includes("exceeded");
                if (isTimeout) {
                    lastMessage.content = `⏱️ 请求超时 — 模型推理时间超出了前端等待上限。后端可能已完成生成，请刷新会话查看。\n\n(${errMsg})`;
                } else {
                    lastMessage.content = `❌ 请求失败: ${errMsg}`;
                }
            }
        } finally {
            loading.value = false;
            streaming.value = false;
        }
    }

    // Clear current session messages
    function clearMessages() {
        messages.value = [];
    }

    // Clear error
    function clearError() {
        error.value = null;
    }

    // 重置所有状态 (用户切换/登出时调用)
    function resetState() {
        sessions.value = [];
        currentSessionId.value = null;
        messages.value = [];
        loading.value = false;
        streaming.value = false;
        error.value = null;
    }

    return {
        sessions,
        currentSessionId,
        messages,
        loading,
        streaming,
        error,
        currentSession,
        loadSessions,
        createSession,
        selectSession,
        deleteSession,
        renameSession,
        sendMessage,
        clearMessages,
        clearError,
        resetState,
    };
});
