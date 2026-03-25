import { useState } from 'react';
import { api } from '@/lib/api';
import type { RebuildResponse } from '@/lib/types';
import { RefreshCw, Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function AdminPage() {
  const [rebuilding, setRebuilding] = useState(false);
  const [result, setResult] = useState<RebuildResponse | null>(null);
  const [error, setError] = useState('');

  const handleRebuild = async () => {
    setRebuilding(true);
    setResult(null);
    setError('');
    try {
      const res = await api.rebuildIndex();
      setResult(res);
    } catch {
      setError('Rebuild failed. Check server logs.');
    } finally {
      setRebuilding(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="px-6 py-4 border-b border-slate-800">
        <h2 className="text-white font-semibold">Admin</h2>
        <p className="text-slate-400 text-xs mt-0.5">System management operations</p>
      </div>

      <div className="px-6 py-6 max-w-2xl space-y-6">
        <section className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-white font-medium">Rebuild Index</h3>
              <p className="text-slate-400 text-sm mt-1">
                Re-processes all documents in <code className="bg-slate-700 text-slate-200 px-1.5 py-0.5 rounded text-xs">data/raw/</code> and rebuilds the FAISS vector index from scratch.
              </p>
              <div className="mt-3 flex items-center gap-2 text-yellow-400 text-xs">
                <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                This operation may take several minutes depending on the number of documents.
              </div>
            </div>
            <button
              onClick={() => void handleRebuild()}
              disabled={rebuilding}
              className="flex items-center gap-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-lg text-sm font-medium transition shrink-0"
            >
              {rebuilding ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
              {rebuilding ? 'Rebuilding…' : 'Rebuild'}
            </button>
          </div>

          {result && (
            <div className="mt-4 bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <div className="flex items-center gap-2 text-green-300 font-medium text-sm mb-2">
                <CheckCircle2 className="w-4 h-4" /> Rebuild Complete
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="text-slate-400 text-xs">Documents</div>
                  <div className="text-white font-semibold text-xl mt-0.5">{result.documents_reindexed}</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-3">
                  <div className="text-slate-400 text-xs">Total Chunks</div>
                  <div className="text-white font-semibold text-xl mt-0.5">{result.chunks_total.toLocaleString()}</div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
              {error}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
