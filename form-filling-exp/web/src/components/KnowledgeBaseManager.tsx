'use client';

import { useState, useCallback, useEffect } from 'react';
import { getKnowledgeBaseDocuments, deleteKnowledgeBaseDocument, streamParseFiles, KnowledgeBaseDocument } from '@/lib/api';
import FileUploadZone, { ParseProgress } from './FileUploadZone';

interface KnowledgeBaseManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function KnowledgeBaseManager({ isOpen, onClose }: KnowledgeBaseManagerProps) {
  const [kbDocuments, setKbDocuments] = useState<KnowledgeBaseDocument[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  
  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [parseProgress, setParseProgress] = useState<ParseProgress | null>(null);

  // Load KB documents when modal opens
  useEffect(() => {
    if (isOpen) {
      loadKbDocuments();
    }
  }, [isOpen]);

  const loadKbDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getKnowledgeBaseDocuments();
      setKbDocuments(result.documents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleUpload = useCallback(async (files: File[]) => {
    setIsUploading(true);
    setParseProgress(null);
    setError(null);

    try {
      for await (const event of streamParseFiles(files, null, false)) {
        if (event.type === 'progress' && event.current !== undefined && event.total !== undefined && event.filename && event.status) {
          setParseProgress({
            current: event.current,
            total: event.total,
            filename: event.filename,
            status: event.status,
            error: event.error,
          });
        }

        if (event.type === 'error' && event.error) {
          setError(event.error);
          return;
        }

        if (event.type === 'complete' && event.results) {
          // Upload complete, refresh the list
          await loadKbDocuments();
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload files');
    } finally {
      setIsUploading(false);
      setParseProgress(null);
    }
  }, [loadKbDocuments]);

  const handleDelete = useCallback(async (documentId: string) => {
    try {
      await deleteKnowledgeBaseDocument(documentId);
      // Refresh the list
      await loadKbDocuments();
      setDeleteConfirm(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document');
    }
  }, [loadKbDocuments]);

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return dateStr;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-background border border-border rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <span className="text-xl">📚</span>
            <h2 className="text-lg font-semibold text-foreground">Knowledge Base Manager</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-background-secondary rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 text-foreground-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Error Message */}
          {error && (
            <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 text-sm">
              {error}
            </div>
          )}

          {/* Upload Section */}
          <div>
            <h3 className="text-sm font-medium text-foreground-secondary mb-3">Upload Documents</h3>
            
            <FileUploadZone
              onUpload={handleUpload}
              isUploading={isUploading}
              parseProgress={parseProgress}
              disabled={false}
              maxFiles={10}
              currentFileCount={0}
              dropZoneSize="large"
              inputId="kb-files-input"
            />

            <p className="mt-2 text-xs text-foreground-muted">
              Documents uploaded here will be available across all sessions as permanent reference material.
            </p>
          </div>

          {/* Documents List Section */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-foreground-secondary">
                All Documents ({kbDocuments.length})
              </h3>
              <button
                onClick={loadKbDocuments}
                disabled={isLoading}
                className="text-xs text-accent hover:text-accent-hover transition-colors disabled:opacity-50"
              >
                {isLoading ? 'Loading...' : 'Refresh'}
              </button>
            </div>

            {isLoading && kbDocuments.length === 0 ? (
              <div className="text-center py-8 text-foreground-muted text-sm">
                Loading documents...
              </div>
            ) : kbDocuments.length === 0 ? (
              <div className="text-center py-8 rounded-lg bg-background-secondary border border-border">
                <svg className="w-12 h-12 mx-auto mb-3 text-foreground-muted opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm text-foreground-muted">No documents in knowledge base</p>
                <p className="text-xs text-foreground-muted mt-1">Upload documents above to get started</p>
              </div>
            ) : (
              <div className="space-y-2">
                {kbDocuments.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between px-4 py-3 rounded-lg bg-background-secondary hover:bg-background-tertiary transition-colors border border-border"
                  >
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <svg className="w-5 h-5 text-accent flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <div className="min-w-0 flex-1">
                        <div className="text-sm text-foreground-secondary font-medium truncate">
                          {doc.filename}
                        </div>
                        <div className="text-xs text-foreground-muted mt-0.5">
                          Uploaded {formatDate(doc.created_at)}
                        </div>
                      </div>
                    </div>

                    {deleteConfirm === doc.id ? (
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <button
                          onClick={() => handleDelete(doc.id)}
                          className="px-3 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
                        >
                          Confirm
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(null)}
                          className="px-3 py-1 text-xs bg-background-tertiary text-foreground-secondary rounded hover:bg-border transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeleteConfirm(doc.id)}
                        className="px-3 py-1 text-xs text-red-500 hover:bg-red-500/10 rounded transition-colors flex-shrink-0"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-border flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-accent text-white hover:bg-accent-hover transition-colors text-sm font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
