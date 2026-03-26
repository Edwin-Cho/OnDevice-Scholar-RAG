import { useState, useRef, useEffect } from 'react';
import { api } from '@/lib/api';
import type { Citation, QueryResponse } from '@/lib/types';
import { Send, Loader2, BookOpen, AlertTriangle, ExternalLink, Copy, Check, Trash2, ChevronDown, ChevronUp } from 'lucide-react';
import MathRenderer from '@/components/MathRenderer';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  status?: QueryResponse['status'];
  warning?: string;
}

const EXAMPLE_QUERIES = [
  'What is the attention mechanism?',
  'How does BERT pre-training work?',
  'Compare encoder-only vs decoder-only models',
  'What is the computational complexity of self-attention?',
];

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? '#6366f1' : score >= 0.65 ? '#4f46e5' : '#4b5563';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] tabular-nums" style={{ color }}>{pct}%</span>
    </div>
  );
}

function CitationCard({ c }: { c: Citation }) {
  return (
    <div className="rounded-xl p-3 text-xs space-y-2 border"
      style={{ background: 'rgba(255,255,255,0.03)', borderColor: 'rgba(255,255,255,0.07)' }}>
      <div className="flex items-start justify-between gap-2">
        <span className="font-medium text-slate-100 leading-snug">{c.paper_title || c.source_filename}</span>
        {c.arxiv_id && (
          <a href={`https://arxiv.org/abs/${c.arxiv_id}`} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-1 text-cyan-400/70 hover:text-cyan-300 transition shrink-0">
            arXiv <ExternalLink className="w-2.5 h-2.5" />
          </a>
        )}
      </div>
      {c.section_header && (
        <div className="text-slate-500 flex items-center gap-1">
          <span className="text-slate-600">§</span> {c.section_header}
        </div>
      )}
      <div className="flex items-center gap-3 text-slate-600">
        <span className="truncate">{c.source_filename}</span>
        {c.page_number && <span className="shrink-0">p.{c.page_number}</span>}
      </div>
      <ScoreBar score={c.score} />
    </div>
  );
}

function AssistantMessage({ msg, onCopy, copied, expanded, onToggleExpand }: {
  msg: Message;
  onCopy: (id: string, text: string) => void;
  copied: boolean;
  expanded: boolean;
  onToggleExpand: (id: string) => void;
}) {
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 bg-indigo-700/50 border border-indigo-600/30 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
        <BookOpen className="w-4 h-4 text-indigo-300" />
      </div>
      <div className="flex-1 space-y-3 min-w-0">
        {msg.warning && (
          <div className="flex items-center gap-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-3 py-2 text-yellow-400 text-xs">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            {msg.warning}
          </div>
        )}
        <div className="relative group">
          <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-4 py-3 text-slate-200 text-sm leading-relaxed whitespace-pre-wrap">
            <MathRenderer text={msg.content} />
          </div>
          <button
            onClick={() => onCopy(msg.id, msg.content)}
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-600/60 text-slate-400 hover:text-slate-200"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
          </button>
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <div>
            <button
              onClick={() => onToggleExpand(msg.id)}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition mb-2"
            >
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              Sources ({msg.citations.length})
            </button>
            {expanded && (
              <div className="grid gap-2">
                {msg.citations.map((c, i) => <CitationCard key={i} c={c} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function QueryPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedCitations, setExpandedCitations] = useState<Record<string, boolean>>({});
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleCopy = (id: string, text: string) => {
    void navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  const handleToggleExpand = (id: string) => {
    setExpandedCitations((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await api.query(q, topK);
      const msgId = (Date.now() + 1).toString();
      const assistantMsg: Message = {
        id: msgId,
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        status: res.status,
        warning: res.warning,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setExpandedCitations((prev) => ({ ...prev, [msgId]: false }));
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: 'assistant', content: 'Error: Failed to get a response. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/60">
        <div>
          <h2 className="text-white font-semibold">Research Query</h2>
          <p className="text-slate-500 text-xs mt-0.5">Ask questions about your indexed papers</p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={() => { setMessages([]); setExpandedCitations({}); }}
            className="flex items-center gap-1.5 text-xs text-slate-600 hover:text-red-400 px-2.5 py-1.5 rounded-lg hover:bg-red-500/8 transition"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16">
            <div className="w-14 h-14 bg-slate-800/80 border border-slate-700/50 rounded-2xl flex items-center justify-center mb-4">
              <BookOpen className="w-7 h-7 text-slate-400" />
            </div>
            <h3 className="text-white font-medium">Ask anything about your papers</h3>
            <p className="text-slate-500 text-sm mt-1.5 max-w-sm">Queries are answered using only your indexed documents with forced citations.</p>
            <div className="flex flex-wrap gap-2 justify-center mt-6 max-w-md">
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => { setInput(q); textareaRef.current?.focus(); }}
                  className="px-3 py-1.5 text-xs rounded-lg border border-slate-700/60 text-slate-400 hover:text-slate-200 hover:border-slate-600 hover:bg-slate-800/60 transition"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id}>
            {msg.role === 'user' ? (
              <div className="flex justify-end">
                <div className="bg-indigo-600/80 border border-indigo-500/30 rounded-xl px-4 py-2.5 text-white text-sm max-w-[70%] whitespace-pre-wrap">
                  {msg.content}
                </div>
              </div>
            ) : (
              <AssistantMessage
                msg={msg}
                onCopy={handleCopy}
                copied={copiedId === msg.id}
                expanded={expandedCitations[msg.id] ?? false}
                onToggleExpand={handleToggleExpand}
              />
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-indigo-700/50 border border-indigo-600/30 rounded-lg flex items-center justify-center shrink-0">
              <BookOpen className="w-4 h-4 text-indigo-300" />
            </div>
            <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-4 py-3 flex items-center gap-2 text-slate-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" /> Retrieving and generating…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="px-6 py-4 border-t border-slate-800/60">
        <form onSubmit={handleSubmit} className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your papers… (Enter to send, Shift+Enter for newline)"
            rows={2}
            className="flex-1 bg-slate-900/80 border border-slate-700/60 rounded-xl px-4 py-3 text-white placeholder-slate-600 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500/60 focus:border-indigo-500/40 resize-none transition"
          />
          <div className="flex flex-col gap-2 shrink-0">
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-2 py-1.5 text-slate-400 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500/60 cursor-pointer"
            >
              {[3, 5, 10, 15].map((v) => <option key={v} value={v}>K={v}</option>)}
            </select>
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="h-9 px-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg flex items-center justify-center transition"
            >
              <Send className="w-4 h-4 text-white" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
