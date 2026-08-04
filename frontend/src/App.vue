<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand-block">
        <div class="brand-kicker">药品说明书 RAG 工作台</div>
        <h1>本地化药品说明书问答系统</h1>
        <p>面向说明书检索、来源核验与中文问答的高密度操作台。</p>
      </div>

      <div class="topbar-tools">
        <div class="api-control">
          <label>API</label>
          <input
            v-model="apiBase"
            type="text"
            spellcheck="false"
            placeholder="http://127.0.0.1:8001"
            @keydown.enter.prevent="connectApi"
          />
          <button class="icon-button primary" :disabled="statsLoading" @click="connectApi">
            <RefreshCw :size="16" />
            <span>{{ statsLoading ? "连接中" : "连接" }}</span>
          </button>
        </div>

        <div class="connection-pill" :class="connectionStateClass">
          <Wifi v-if="statsOk" :size="15" />
          <WifiOff v-else :size="15" />
          <span>{{ connectionLabel }}</span>
        </div>
      </div>
    </header>

    <main class="workspace">
      <aside class="sidebar">
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">
              <Database :size="18" />
              <span>系统状态</span>
            </div>
            <button class="ghost-button" :disabled="statsLoading" @click="refreshStats">
              <Activity :size="14" />
              <span>刷新</span>
            </button>
          </div>

          <div v-if="stats" class="stat-grid">
            <div class="stat-item">
              <span>索引片段</span>
              <strong>{{ formatCount(stats.documents) }}</strong>
            </div>
            <div class="stat-item">
              <span>知识库</span>
              <strong>{{ stats.collection }}</strong>
            </div>
            <div class="stat-item">
              <span>检索链路</span>
              <strong>{{ stats.retrieval }}</strong>
            </div>
            <div class="stat-item">
              <span>重排模型</span>
              <strong>{{ compactModel(stats.rerank) }}</strong>
            </div>
            <div class="stat-item">
              <span>生成模型</span>
              <strong>{{ stats.llm }}</strong>
            </div>
            <div class="stat-item">
              <span>查询设备</span>
              <strong>{{ stats.query_embedding_device }}</strong>
            </div>
          </div>

          <div v-else class="empty-state">
            <p>{{ statsError || "正在拉取后端状态。" }}</p>
          </div>

          <div v-if="stats" class="notice-row">
            <span class="notice-tag success">Milvus</span>
            <span class="notice-tag">BM25</span>
            <span class="notice-tag">Rerank</span>
            <span class="notice-tag">Ollama</span>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">
              <Upload :size="18" />
              <span>导入文件</span>
            </div>
          </div>

          <input ref="fileInput" class="file-input" type="file" @change="onFileChange" />
          <div class="upload-zone" @click="openFilePicker">
            <FileText :size="22" />
            <div>
              <strong>点击选择 PDF / Excel / TXT / CSV</strong>
              <p>{{ selectedFileName || "支持说明书批量导入，自动切分并入库。" }}</p>
            </div>
          </div>

          <div class="action-row">
            <button class="icon-button" :disabled="!selectedFile || uploadLoading" @click="uploadFile">
              <Upload :size="16" />
              <span>{{ uploadLoading ? "写入中" : "写入知识库" }}</span>
            </button>
            <button class="ghost-button" :disabled="!selectedFile" @click="clearSelectedFile">
              <X :size="14" />
              <span>清除</span>
            </button>
          </div>

          <div v-if="uploadResult" class="result-box" :class="{ error: uploadResult.error }">
            <div class="result-line">
              <span>状态</span>
              <strong>{{ uploadResult.error ? "失败" : uploadResult.status || "success" }}</strong>
            </div>
            <div class="result-line">
              <span>文件</span>
              <strong>{{ uploadResult.filename || selectedFileName || "-" }}</strong>
            </div>
            <div class="result-line">
              <span>写入</span>
              <strong>{{ uploadResult.chunks_count ?? 0 }}</strong>
            </div>
            <p v-if="uploadResult.error">{{ uploadResult.error }}</p>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">
              <Sparkles :size="18" />
              <span>常用问题</span>
            </div>
          </div>

          <div class="question-list">
            <button
              v-for="item in presetQuestions"
              :key="item"
              class="question-chip"
              @click="applyQuestion(item)"
            >
              {{ item }}
            </button>
          </div>
        </section>
      </aside>

      <section class="center">
        <section class="panel chat-panel">
          <div class="panel-head">
            <div class="panel-title">
              <MessageSquareMore :size="18" />
              <span>问答对话</span>
            </div>
            <div class="panel-meta">
              <span>{{ messages.length }} 条消息</span>
              <span v-if="currentAnswerMeta">最近一次 {{ currentAnswerMeta.responseTime }}s</span>
            </div>
          </div>

          <div ref="chatScrollRef" class="chat-list">
            <div v-if="messages.length === 0" class="empty-chat">
              <h2>先问一个药品问题</h2>
              <p>例如：阿司匹林肠溶片的禁忌是什么，或头孢克肟片成人怎么服用。</p>
            </div>

            <article v-for="message in messages" :key="message.id" class="message" :class="message.role">
              <div class="message-head">
                <div class="message-role">
                  <User v-if="message.role === 'user'" :size="15" />
                  <Bot v-else :size="15" />
                  <span>{{ message.role === "user" ? "提问" : "回答" }}</span>
                </div>
                <div v-if="message.role === 'assistant'" class="message-summary">
                  <span>置信度 {{ formatPercent(message.meta.confidence) }}</span>
                  <span>支持度 {{ formatPercent(message.verification.support_ratio) }}</span>
                </div>
              </div>

              <div class="bubble">{{ message.content }}</div>

              <div v-if="message.role === 'assistant'" class="assistant-footer">
                <span>来源 {{ message.sources.length }}</span>
                <span>耗时 {{ message.meta.responseTime }}s</span>
                <span :class="['verification-badge', verificationTone(message.verification)]">
                  {{ verificationLabel(message.verification) }}
                </span>
              </div>
            </article>
          </div>

          <div class="composer">
            <textarea
              v-model="question"
              rows="4"
              placeholder="请输入药品说明书相关问题，Shift + Enter 可换行。"
              @keydown.enter.exact.prevent="sendQuestion"
            />

            <div class="composer-actions">
              <div class="composer-hint">
                <span>支持来源展示、验证结果、置信度和响应时间。</span>
              </div>

              <div class="action-row">
                <button class="ghost-button" :disabled="sendLoading || messages.length === 0" @click="clearChat">
                  <Trash2 :size="14" />
                  <span>清空</span>
                </button>
                <button class="icon-button primary" :disabled="sendLoading || !question.trim()" @click="sendQuestion">
                  <Send :size="16" />
                  <span>{{ sendLoading ? "生成中" : "发送问题" }}</span>
                </button>
              </div>
            </div>
          </div>
        </section>
      </section>

      <aside class="sidebar right">
        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">
              <ShieldCheck :size="18" />
              <span>来源与验证</span>
            </div>
          </div>

          <div v-if="currentAnswerMeta" class="evidence-summary">
            <div class="summary-line">
              <span>置信度</span>
              <strong>{{ formatPercent(currentAnswerMeta.confidence) }}</strong>
            </div>
            <div class="summary-line">
              <span>支持率</span>
              <strong>{{ formatPercent(currentVerification.support_ratio) }}</strong>
            </div>
            <div class="summary-line">
              <span>响应时间</span>
              <strong>{{ currentAnswerMeta.responseTime }}s</strong>
            </div>
            <p class="verification-note">{{ currentVerification.verification_note || "等待新的回答结果。" }}</p>
          </div>

          <div v-else class="empty-state">
            <p>回答后会在这里显示来源片段与验证结果。</p>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <div class="panel-title">
              <FileText :size="18" />
              <span>Top 来源</span>
            </div>
          </div>

          <div v-if="currentSources.length" class="source-list">
            <article v-for="source in currentSources" :key="`${source.index}-${source.source}`" class="source-item">
              <div class="source-head">
                <strong>{{ source.index }}. {{ source.source }}</strong>
                <span>{{ formatScore(source.score) }}</span>
              </div>
              <div class="source-meta">
                <span v-if="source.page">{{ source.page }}</span>
                <span v-if="source.section">{{ source.section }}</span>
              </div>
              <p>{{ source.text }}</p>
            </article>
          </div>

          <div v-else class="empty-state">
            <p>暂无来源，先发起一次问答。</p>
          </div>
        </section>
      </aside>
    </main>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from "vue";
import {
  Activity,
  Bot,
  Database,
  FileText,
  MessageSquareMore,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  User,
  Wifi,
  WifiOff,
  X,
} from "lucide-vue-next";
import { askQuestion, fetchStats, getSavedApiBase, saveApiBase, uploadDocument } from "./api";

const apiBase = ref(getSavedApiBase());
const stats = ref(null);
const statsLoading = ref(false);
const statsError = ref("");
const messages = ref([]);
const question = ref("");
const sendLoading = ref(false);
const selectedFile = ref(null);
const uploadLoading = ref(false);
const uploadResult = ref(null);
const fileInput = ref(null);
const chatScrollRef = ref(null);

const presetQuestions = [
  "阿司匹林肠溶片的禁忌是什么？",
  "头孢克肟片成人一般怎么服用？",
  "氯雷他定片可能有哪些不良反应？",
  "蒙脱石散适合哪些腹泻情况？",
  "奥美拉唑肠溶胶囊适应症是什么？",
  "阿奇霉素片可能有哪些胃肠道不良反应？",
];

const statsOk = computed(() => Boolean(stats.value && !statsError.value));
const connectionStateClass = computed(() => {
  if (statsOk.value) return "ok";
  if (statsError.value) return "error";
  return "idle";
});
const connectionLabel = computed(() => {
  if (statsOk.value) return "后端已连接";
  if (statsError.value) return "后端异常";
  return "等待连接";
});
const selectedFileName = computed(() => selectedFile.value?.name || "");
const lastAssistantMessage = computed(() => {
  return [...messages.value].reverse().find((item) => item.role === "assistant");
});
const currentSources = computed(() => lastAssistantMessage.value?.sources || []);
const currentVerification = computed(() => {
  return lastAssistantMessage.value?.verification || {
    verified: false,
    support_level: "none",
    support_ratio: 0,
    verification_note: "",
  };
});
const currentAnswerMeta = computed(() => lastAssistantMessage.value?.meta || null);

onMounted(() => {
  refreshStats();
});

function connectApi() {
  saveApiBase(apiBase.value);
  refreshStats();
}

async function refreshStats() {
  statsLoading.value = true;
  statsError.value = "";
  try {
    stats.value = await fetchStats(apiBase.value);
  } catch (error) {
    stats.value = null;
    statsError.value = error.message || "无法连接后端";
  } finally {
    statsLoading.value = false;
  }
}

function openFilePicker() {
  fileInput.value?.click();
}

function onFileChange(event) {
  selectedFile.value = event.target.files?.[0] || null;
  uploadResult.value = null;
}

function clearSelectedFile() {
  selectedFile.value = null;
  uploadResult.value = null;
  if (fileInput.value) {
    fileInput.value.value = "";
  }
}

async function uploadFile() {
  if (!selectedFile.value || uploadLoading.value) return;
  uploadLoading.value = true;
  uploadResult.value = null;
  try {
    uploadResult.value = await uploadDocument(apiBase.value, selectedFile.value);
    await refreshStats();
  } catch (error) {
    uploadResult.value = { error: error.message || "上传失败" };
  } finally {
    uploadLoading.value = false;
  }
}

function applyQuestion(value) {
  question.value = value;
}

async function sendQuestion() {
  const text = question.value.trim();
  if (!text || sendLoading.value) return;

  messages.value.push({
    id: `user-${Date.now()}`,
    role: "user",
    content: text,
  });
  question.value = "";
  sendLoading.value = true;
  await scrollChat();

  try {
    const payload = await askQuestion(apiBase.value, text);
    messages.value.push({
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: payload.answer || "未返回有效答案。",
      sources: Array.isArray(payload.sources) ? payload.sources : [],
      verification: payload.verification || {},
      meta: {
        confidence: Number(payload.confidence || 0),
        responseTime: formatSeconds(payload.response_time || 0),
      },
    });
  } catch (error) {
    messages.value.push({
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: `问答失败：${error.message || error}`,
      sources: [],
      verification: {
        verified: false,
        support_level: "low",
        support_ratio: 0,
        verification_note: "请求失败，未生成可验证答案。",
      },
      meta: {
        confidence: 0,
        responseTime: "0.00",
      },
    });
  } finally {
    sendLoading.value = false;
    await scrollChat();
  }
}

function clearChat() {
  messages.value = [];
}

async function scrollChat() {
  await nextTick();
  if (chatScrollRef.value) {
    chatScrollRef.value.scrollTop = chatScrollRef.value.scrollHeight;
  }
}

function compactModel(value) {
  if (!value) return "-";
  if (value === "disabled") return "关闭";
  const normalized = String(value).replaceAll("\\", "/");
  if (normalized.includes("/snapshots/")) {
    const [modelRoot] = normalized.split("/snapshots/");
    return modelRoot.split("/").pop().replace("models--", "").replaceAll("--", "/");
  }
  return normalized.split("/").pop();
}

function formatCount(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

function formatPercent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function formatSeconds(value) {
  return Number(value || 0).toFixed(2);
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function verificationLabel(value) {
  if (value?.verified && value?.support_level === "high") return "高支持";
  if (value?.verified) return "已验证";
  if (value?.support_level === "review") return "建议复核";
  return "低支持";
}

function verificationTone(value) {
  if (value?.verified && value?.support_level === "high") return "high";
  if (value?.verified) return "medium";
  if (value?.support_level === "review") return "review";
  return "low";
}
</script>
