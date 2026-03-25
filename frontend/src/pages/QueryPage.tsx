import { useState, useRef, useEffect } from 'react';
import { api } from '@/lib/api';
import type { Citation, QueryResponse } from '@/lib/types';
import { Send, Loader2, BookOpen, AlertTriangle, ExternalLink } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  status?: QueryResponse['status'];
  warning?: string;
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? '#a855f7' : score >= 0.65 ? '#6366f1' : '#64748b';
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
        <span className="font-medium text-purple-300 leading-snug">{c.paper_title || c.source_filename}</span>
        {c.arxiv_id && (
          <a href={`https://arxiv.org/abs/${c.arxiv_id}`} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-1 text-purple-400/70 hover:text-purple-300 transition shrink-0">
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

function AssistantMessage({ msg }: { msg: Message }) {
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center shrink-0 mt-0.5">
        <BookOpen className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 space-y-3 min-w-0">
        {msg.warning && (
          <div className="flex items-center gap-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-3 py-2 text-yellow-400 text-xs">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            {msg.warning}
          </div>
        )}
        <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-4 py-3 text-slate-200 text-sm leading-relaxed whitespace-pre-wrap">
          {msg.content}
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <div>
            <p className="text-xs text-slate-500 mb-2">Sources ({msg.citations.length})</p>
            <div className="grid gap-2">
              {msg.citations.map((c, i) => <CitationCard key={i} c={c} />)}
            </div>
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
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: res.answer,
        citations: res.citations,
        status: res.status,
        warning: res.warning,
      };
      setMessages((prev) => [...prev, assistantMsg]);
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
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
        <div>
          <h2 className="text-white font-semibold">Research Query</h2>
          <p className="text-slate-400 text-xs mt-0.5">Ask questions about your indexed papers</p>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span>Top-K</span>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-white focus:outline-none focus:ring-1 focus:ring-purple-500"
          >
            {[3, 5, 10, 15].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-20">
            <div className="w-16 h-16 bg-purple-600/20 border border-purple-500/30 rounded-2xl flex items-center justify-center mb-4">
              <BookOpen className="w-8 h-8 text-purple-400" />
            </div>
            <h3 className="text-white font-medium text-lg">Ask anything about your papers</h3>
            <p className="text-slate-400 text-sm mt-2 max-w-sm">Queries are answered using only your indexed documents with forced citations.</p>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id}>
            {msg.role === 'user' ? (
              <div className="flex justify-end">
                <div className="bg-purple-600 rounded-xl px-4 py-2.5 text-white text-sm max-w-[70%] whitespace-pre-wrap">
                  {msg.content}
                </div>
              </div>
            ) : (
              <AssistantMessage msg={msg} />
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center shrink-0">
              <BookOpen className="w-4 h-4 text-white" />
            </div>
            <div className="bg-slate-800/60 border border-slate-700/40 rounded-xl px-4 py-3 flex items-center gap-2 text-slate-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" /> Retrieving and generating…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="px-6 py-4 border-t border-slate-800">
        <form onSubmit={handleSubmit} className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your papers… (Enter to send, Shift+Enter for newline)"
            rows={2}
            className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="w-10 h-10 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition shrink-0"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </form>
      </div>
    </div>
  );
}
