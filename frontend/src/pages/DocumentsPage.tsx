import { useState, useRef } from 'react';
import { api } from '@/lib/api';
import { Upload, Trash2, FileText, Loader2, CheckCircle2, XCircle } from 'lucide-react';

interface UploadResult {
  filename: string;
  status: 'success' | 'error';
  message: string;
}

export default function DocumentsPage() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadResult[]>([]);
  const [deleteTarget, setDeleteTarget] = useState('');
  const [deleting, setDeleting] = useState(false);
  const [deleteResult, setDeleteResult] = useState<{ status: 'success' | 'error'; message: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) =>
      ['application/pdf', 'text/plain', 'text/markdown'].includes(f.type) ||
      f.name.endsWith('.md') || f.name.endsWith('.txt') || f.name.endsWith('.pdf')
    );
    if (!arr.length) return;
    setUploading(true);
    setResults([]);
    for (const file of arr) {
      try {
        const res = await api.ingest(file);
        setResults((prev) => [...prev, { filename: file.name, status: 'success', message: `${res.chunks_added} chunks indexed` }]);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        setResults((prev) => [...prev, { filename: file.name, status: 'error', message: msg }]);
      }
    }
    setUploading(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void uploadFiles(e.dataTransfer.files);
  };

  const handleDelete = async () => {
    const name = deleteTarget.trim();
    if (!name) return;
    setDeleting(true);
    setDeleteResult(null);
    try {
      const res = await api.deleteDocument(name);
      setDeleteResult({ status: 'success', message: `Removed ${res.chunks_removed} chunks` });
      setDeleteTarget('');
    } catch {
      setDeleteResult({ status: 'error', message: 'Delete failed — check filename and try again' });
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="px-6 py-4 border-b border-slate-800">
        <h2 className="text-white font-semibold">Documents</h2>
        <p className="text-slate-400 text-xs mt-0.5">Upload and manage indexed papers</p>
      </div>

      <div className="px-6 py-6 space-y-8 max-w-2xl">
        {/* Upload Zone */}
        <section>
          <h3 className="text-sm font-medium text-slate-300 mb-3">Upload Documents</h3>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors ${
              dragging ? 'border-purple-500 bg-purple-500/10' : 'border-slate-700 hover:border-purple-600 hover:bg-slate-800/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.md"
              className="hidden"
              onChange={(e) => e.target.files && void uploadFiles(e.target.files)}
            />
            <Upload className="w-10 h-10 text-slate-500 mx-auto mb-3" />
            <p className="text-slate-300 font-medium">Drop files here or click to browse</p>
            <p className="text-slate-500 text-sm mt-1">Supported: PDF, TXT, Markdown</p>
          </div>

          {uploading && (
            <div className="flex items-center gap-2 mt-4 text-purple-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" /> Uploading and indexing…
            </div>
          )}

          {results.length > 0 && (
            <div className="mt-4 space-y-2">
              {results.map((r, i) => (
                <div
                  key={i}
                  className={`flex items-center gap-3 rounded-lg px-4 py-2.5 text-sm ${
                    r.status === 'success'
                      ? 'bg-green-500/10 border border-green-500/30 text-green-300'
                      : 'bg-red-500/10 border border-red-500/30 text-red-300'
                  }`}
                >
                  {r.status === 'success' ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <XCircle className="w-4 h-4 shrink-0" />}
                  <span className="font-medium truncate">{r.filename}</span>
                  <span className="text-xs opacity-70 ml-auto shrink-0">{r.message}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Delete Section */}
        <section>
          <h3 className="text-sm font-medium text-slate-300 mb-3">Delete Document</h3>
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 space-y-3">
            <div className="flex items-center gap-2 text-slate-400 text-xs">
              <FileText className="w-4 h-4" />
              Enter the exact filename (e.g. <code className="bg-slate-700 px-1.5 py-0.5 rounded text-slate-300">BERT.pdf</code>)
            </div>
            <div className="flex gap-3">
              <input
                type="text"
                value={deleteTarget}
                onChange={(e) => setDeleteTarget(e.target.value)}
                placeholder="filename.pdf"
                className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent"
              />
              <button
                onClick={() => void handleDelete()}
                disabled={!deleteTarget.trim() || deleting}
                className="flex items-center gap-2 bg-red-600/80 hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm transition"
              >
                {deleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                Delete
              </button>
            </div>
            {deleteResult && (
              <div className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                deleteResult.status === 'success'
                  ? 'bg-green-500/10 border border-green-500/30 text-green-300'
                  : 'bg-red-500/10 border border-red-500/30 text-red-300'
              }`}>
                {deleteResult.status === 'success' ? <CheckCircle2 className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                {deleteResult.message}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
