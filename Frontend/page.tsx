"use client";

import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Link from 'next/link';
import { BrainCircuit, Plus, Book, Bell, Clock, Database, X, Trash2 } from 'lucide-react';

interface HistoryItem {
  id: string;
  query: string;
  mode: string;
  report: string;
  date: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedReport, setSelectedReport] = useState<HistoryItem | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      const saved = localStorage.getItem('agenticHistory');
      if (saved) {
        setHistory(JSON.parse(saved));
      }
    }, 0);

    return () => clearTimeout(timer);
  }, []);

  const clearHistory = () => {
    if(confirm("Are you sure you want to clear your research history?")) {
      localStorage.removeItem('agenticHistory');
      setHistory([]);
    }
  }

  // Individual delete function
  const deleteHistoryItem = (e: React.MouseEvent, idToDelete: string) => {
    e.stopPropagation(); // Prevents the card click event from opening the modal
    if(confirm("Delete this report from your archive?")) {
      const updatedHistory = history.filter(item => item.id !== idToDelete);
      setHistory(updatedHistory);
      localStorage.setItem('agenticHistory', JSON.stringify(updatedHistory));
    }
  };

  const navItemClass = 'flex items-center gap-3 px-4 py-2.5 text-slate-600 hover:text-[#0F172A] hover:bg-slate-100 rounded-xl font-medium text-sm transition-colors';

  return (
    <div className="flex h-screen bg-white text-[#0F172A] font-sans overflow-hidden">

      {/* SIDEBAR \u2014 matches the main layout: white surface, navy typography, teal active state */}
      <aside className="w-72 bg-slate-50 border-r border-slate-200 p-5 flex flex-col shrink-0 relative z-20">
        <div className="flex items-center gap-3 mb-6">
          <BrainCircuit className="w-7 h-7 text-teal-600" />
          <h1 className="text-xl font-bold text-[#0F172A] tracking-tight">A.R.I.A.</h1>
        </div>

        <nav className="flex flex-col gap-1">
          <Link
            href="/"
            className="flex items-center gap-3 px-4 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl font-semibold text-sm transition-colors"
          >
            <Plus className="w-4 h-4" /> New
          </Link>
          <Link href="/" className={navItemClass}>
            <Clock className="w-4 h-4" /> Recents
          </Link>
          {/* Active item: teal highlighting */}
          <Link
            href="/history"
            className="flex items-center gap-3 px-4 py-2.5 bg-teal-50 text-teal-700 rounded-xl font-medium text-sm border border-teal-200"
          >
            <Book className="w-4 h-4" /> Library
          </Link>
          <button className={`${navItemClass} cursor-default text-left`} title="Coming soon">
            <Bell className="w-4 h-4" /> Alerts
            <span className="ml-auto text-[9px] bg-slate-200 text-slate-500 rounded-full px-1.5 py-0.5 font-bold">SOON</span>
          </button>
        </nav>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 overflow-y-auto p-10 relative bg-slate-50/60">
        <div className="max-w-6xl mx-auto">

          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-semibold text-[#0F172A] tracking-tight">Research Library</h2>
              <p className="text-slate-500 mt-1">Your localized archive of AI-generated reports.</p>
            </div>
            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors border border-red-200"
              >
                Clear Archive
              </button>
            )}
          </div>

          {/* EMPTY STATE */}
          {history.length === 0 && (
            <div className="text-center mt-32 border-2 border-dashed border-slate-200 rounded-2xl p-12 bg-white">
              <Database className="w-12 h-12 text-slate-300 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-[#0F172A]">No records found</h3>
              <p className="text-slate-500 mt-2">Run a research query on the dashboard to populate your library.</p>
            </div>
          )}

          {/* HISTORY GRID \u2014 clean, elevated white cards with subtle borders */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {history.map((item) => (
              <div
                key={item.id}
                onClick={() => setSelectedReport(item)}
                className="bg-white border border-slate-200 p-6 rounded-2xl cursor-pointer shadow-sm hover:border-teal-400 hover:shadow-md transition-all group relative"
              >
                <div className="flex items-center gap-2 text-xs text-slate-400 mb-3">
                  <Clock className="w-3 h-3" />
                  <span>{item.date}</span>
                  <span className="ml-auto px-3 py-1 bg-teal-50 text-teal-700 border border-teal-200 rounded-full font-medium whitespace-nowrap flex items-center justify-center">
                    {item.mode}
                  </span>

                  {/* Delete Button */}
                  <button
                    onClick={(e) => deleteHistoryItem(e, item.id)}
                    className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors ml-1"
                    title="Delete report"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <h3 className="text-lg font-semibold text-[#0F172A] group-hover:text-teal-700 transition-colors line-clamp-2">
                  {item.query}
                </h3>
                <p className="text-sm text-slate-500 mt-3 line-clamp-3">
                  {item.report.replace(/[#*]/g, '')}
                </p>
              </div>
            ))}
          </div>

        </div>
      </main>

      {/* REPORT VIEW MODAL */}
      {selectedReport && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-10 bg-slate-900/40 backdrop-blur-sm">
          <div className="bg-white border border-slate-200 w-full max-w-4xl max-h-full rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">

            <div className="flex items-center justify-between p-6 border-b border-slate-200 bg-slate-50">
              <div>
                <h3 className="text-xl font-semibold text-[#0F172A] line-clamp-1">{selectedReport.query}</h3>
                <p className="text-sm text-slate-500 mt-1">Archived on {selectedReport.date}</p>
              </div>
              <button
                onClick={() => setSelectedReport(null)}
                className="p-2 text-slate-400 hover:text-[#0F172A] hover:bg-slate-100 rounded-full transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <div className="p-8 overflow-y-auto bg-white">
              <div className="prose prose-slate max-w-none prose-headings:text-[#0F172A] prose-headings:border-b prose-headings:border-slate-200 prose-headings:pb-2 prose-a:text-teal-700 hover:prose-a:text-teal-800">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {selectedReport.report}
                </ReactMarkdown>
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
