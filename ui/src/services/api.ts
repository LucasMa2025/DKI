import axios, { AxiosInstance, AxiosError } from "axios";
import type {
    User,
    LoginRequest,
    LoginResponse,
    ChatRequest,
    ChatResponse,
    Session,
    ChatMessage,
    UserPreference,
    SystemStats,
} from "@/types";
import config from "@/config";

// Create axios instance
const http: AxiosInstance = axios.create({
    baseURL: config.api.baseUrl,
    timeout: config.api.timeout,
    headers: {
        "Content-Type": "application/json",
    },
});

// Request interceptor
http.interceptors.request.use(
    (config) => {
        // Get token from localStorage
        const authData = localStorage.getItem("auth");
        if (authData) {
            try {
                const { token } = JSON.parse(authData);
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
            } catch {
                // Ignore parse errors
            }
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor
http.interceptors.response.use(
    (response) => response.data,
    (error: AxiosError) => {
        if (error.response?.status === 401) {
            // Clear auth data and redirect to login
            localStorage.removeItem("auth");
            window.location.href = "/login";
        }

        const message =
            (error.response?.data as { detail?: string })?.detail ||
            error.message;
        return Promise.reject(new Error(message));
    }
);

// API methods
export const api = {
    // Auth - 使用 /api/auth 前缀
    auth: {
        async login(credentials: LoginRequest): Promise<LoginResponse> {
            return http.post("/api/auth/login", credentials);
        },

        async logout(): Promise<void> {
            return http.post("/api/auth/logout");
        },

        async getCurrentUser(): Promise<User> {
            return http.get("/api/auth/me");
        },

        async register(data: {
            username: string;
            password: string;
            email?: string;
        }): Promise<User> {
            return http.post("/api/auth/register", data);
        },
    },

    // Chat
    // 修正: 使用 DKI 插件 API
    // DKI 会自动处理偏好读取、历史检索和注入
    chat: {
        async send(request: ChatRequest): Promise<ChatResponse> {
            // 使用 DKI 插件 API: /v1/dki/chat
            // 只传递:
            // - query: 原始用户输入 (不含任何 prompt 构造)
            // - user_id: 用户标识 (DKI 用于读取偏好和历史)
            // - session_id: 会话标识 (DKI 用于读取会话历史)
            return http.post("/v1/dki/chat", {
                // 原始用户输入，不拼接任何历史或 prompt
                query: request.query,
                // 用户标识 - DKI 用于读取偏好和历史
                user_id: request.dkiUserId,
                // 会话标识 - DKI 用于读取会话历史
                session_id: request.dkiSessionId,
                // 可选参数
                model: request.model,
                temperature: request.temperature,
                max_tokens: request.maxTokens,
            });
        },

        // Streaming chat (returns EventSource)
        createStream(request: ChatRequest): EventSource {
            const params = new URLSearchParams({
                query: request.query || "",
                user_id: request.dkiUserId || "",
                session_id: request.dkiSessionId || "",
                model: request.model || "",
                temperature: String(request.temperature || 0.7),
                max_tokens: String(request.maxTokens || 2048),
            });

            return new EventSource(
                `${config.api.baseUrl}/v1/dki/chat/stream?${params}`
            );
        },
    },

    // Sessions - 使用 /api/sessions 前缀
    sessions: {
        async list(): Promise<Session[]> {
            return http.get("/api/sessions");
        },

        async get(id: string): Promise<Session> {
            return http.get(`/api/sessions/${id}`);
        },

        async create(title: string): Promise<Session> {
            return http.post("/api/sessions", { title });
        },

        async update(id: string, data: Partial<Session>): Promise<Session> {
            return http.patch(`/api/sessions/${id}`, data);
        },

        async delete(id: string): Promise<void> {
            return http.delete(`/api/sessions/${id}`);
        },

        async getMessages(sessionId: string): Promise<ChatMessage[]> {
            return http.get(`/api/sessions/${sessionId}/messages`);
        },
    },

    // Preferences - 使用 /api/preferences 前缀
    preferences: {
        async list(userId: string): Promise<UserPreference[]> {
            const response = await http.get(`/api/preferences`, {
                params: { user_id: userId },
            });
            // 后端返回 camelCase 字段，直接返回
            return response as UserPreference[];
        },

        async get(id: string): Promise<UserPreference> {
            return http.get(`/api/preferences/${id}`);
        },

        async create(
            preference: Omit<UserPreference, "id" | "createdAt" | "updatedAt">
        ): Promise<UserPreference> {
            return http.post("/api/preferences", {
                user_id: preference.userId,
                preference_text: preference.preferenceText,
                preference_type: preference.preferenceType,
                priority: preference.priority,
                category: preference.category,
                metadata: preference.metadata,
                is_active: preference.isActive,
            });
        },

        async update(
            id: string,
            updates: Partial<UserPreference>
        ): Promise<UserPreference> {
            return http.patch(`/api/preferences/${id}`, {
                preference_text: updates.preferenceText,
                preference_type: updates.preferenceType,
                priority: updates.priority,
                category: updates.category,
                metadata: updates.metadata,
                is_active: updates.isActive,
            });
        },

        async delete(id: string): Promise<void> {
            return http.delete(`/api/preferences/${id}`);
        },
    },

    // Stats - 使用 /api/stats 前缀
    stats: {
        async getSystemStats(): Promise<SystemStats> {
            return http.get("/api/stats");
        },

        async getDKIStats(): Promise<SystemStats["dkiStats"]> {
            return http.get("/api/stats/dki");
        },

        async getCacheStats(): Promise<SystemStats["cacheStats"]> {
            return http.get("/api/stats/cache");
        },
    },

    // Health - 使用 /api/health
    health: {
        async check(): Promise<{ status: string; version: string }> {
            return http.get("/api/health");
        },
    },

    // Models - 使用 /v1/models (OpenAI 兼容)
    models: {
        async list(): Promise<{
            object: string;
            data: Array<{
                id: string;
                object: string;
                created: number;
                owned_by: string;
                permission: any[];
                root: string;
                parent: string | null;
            }>;
        }> {
            return http.get("/v1/models");
        },
    },

    // Visualization
    visualization: {
        async getLatest(): Promise<any> {
            return http.get("/v1/dki/visualization/latest");
        },

        async getHistory(
            page: number = 1,
            pageSize: number = 20
        ): Promise<{
            items: any[];
            total: number;
            page: number;
            page_size: number;
        }> {
            return http.get("/v1/dki/visualization/history", {
                params: { page, page_size: pageSize },
            });
        },

        async getDetail(requestId: string): Promise<any> {
            return http.get(`/v1/dki/visualization/detail/${requestId}`);
        },

        async getFlowDiagram(): Promise<any> {
            return http.get("/v1/dki/visualization/flow-diagram");
        },

        async clearHistory(): Promise<{ message: string; success: boolean }> {
            return http.delete("/v1/dki/visualization/history");
        },
    },
};

export default api;
