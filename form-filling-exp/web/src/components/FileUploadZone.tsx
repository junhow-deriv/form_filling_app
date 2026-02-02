'use client';

import { useCallback, useState } from 'react';
import { FileIcon } from './FileIcon';

export interface ParseProgress {
  current: number;
  total: number;
  filename: string;
  status: 'parsing' | 'reading_text' | 'docling' | 'complete' | 'error';
  error?: string;
}

interface FileUploadZoneProps {
  onUpload: (files: File[]) => Promise<void>;
  isUploading: boolean;
  parseProgress: ParseProgress | null;
  disabled?: boolean;
  maxFiles?: number;
  currentFileCount?: number; // For components that track uploaded files separately
  dropZoneSize?: 'small' | 'large'; // Small for ContextFilesUpload, large for KnowledgeBaseManager
  inputId: string; // Unique ID for the file input
}

export default function FileUploadZone({
  onUpload,
  isUploading,
  parseProgress,
  disabled = false,
  maxFiles = 10,
  currentFileCount = 0,
  dropZoneSize = 'small',
  inputId,
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled && !isUploading) setIsDragging(true);
  }, [disabled, isUploading]);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled || isUploading) return;

    const droppedFiles = Array.from(e.dataTransfer.files);
    const totalCount = currentFileCount + pendingFiles.length;
    const remainingSlots = maxFiles - totalCount;

    if (remainingSlots <= 0) return;

    const newFiles = droppedFiles.slice(0, remainingSlots);
    setPendingFiles(prev => [...prev, ...newFiles]);
  }, [disabled, isUploading, currentFileCount, pendingFiles.length, maxFiles]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;

    const selectedFiles = Array.from(e.target.files);
    const totalCount = currentFileCount + pendingFiles.length;
    const remainingSlots = maxFiles - totalCount;

    if (remainingSlots <= 0) return;

    const newFiles = selectedFiles.slice(0, remainingSlots);
    setPendingFiles(prev => [...prev, ...newFiles]);

    // Reset the input
    e.target.value = '';
  }, [currentFileCount, pendingFiles.length, maxFiles]);

  const handleRemovePending = useCallback((index: number) => {
    setPendingFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleUploadClick = useCallback(async () => {
    if (pendingFiles.length === 0 || isUploading) return;

    await onUpload(pendingFiles);
    setPendingFiles([]);
  }, [pendingFiles, isUploading, onUpload]);

  const totalCount = currentFileCount + pendingFiles.length;
  const canAddMore = totalCount < maxFiles;

  const isSmallSize = dropZoneSize === 'small';

  return (
    <>
      {/* Drop zone - only show if can add more */}
      {canAddMore && (
        <div
          className={`
            relative border-2 border-dashed rounded-lg text-center cursor-pointer transition-all
            ${isSmallSize ? 'p-4' : 'p-6'}
            ${isDragging ? 'border-accent bg-accent/10' : 'border-border hover:border-accent/50'}
            ${disabled || isUploading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && !isUploading && document.getElementById(inputId)?.click()}
        >
          <input
            id={inputId}
            type="file"
            multiple
            accept=".pdf,.pptx,.ppt,.docx,.doc,.xlsx,.xls,.png,.jpg,.jpeg,.gif,.txt,.md,.csv,.json,.xml,.html"
            className="hidden"
            onChange={handleFileChange}
            disabled={disabled || isUploading}
          />

          <div className="flex flex-col items-center gap-2">
            <svg 
              className={`text-foreground-muted ${isSmallSize ? 'w-8 h-8' : 'w-10 h-10'}`}
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className={`text-foreground-secondary ${isSmallSize ? 'text-xs' : 'text-sm'}`}>
              Drop files or <span className={`text-accent ${isSmallSize ? '' : 'font-medium'}`}>browse</span>
            </p>
            <p className="text-xs text-foreground-muted">
              PDF, PPTX, DOCX, images, or text files
            </p>
          </div>
        </div>
      )}

      {/* Pending files list */}
      {pendingFiles.length > 0 && (
        <div className={`space-y-2 ${canAddMore ? 'mt-3' : ''}`}>
          <p className="text-xs text-foreground-muted">Ready to upload:</p>
          <div className="space-y-1">
            {pendingFiles.map((file, index) => (
              <div
                key={`pending-${index}`}
                className="flex items-center justify-between px-3 py-2 rounded-lg bg-background-tertiary"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <FileIcon filename={file.name} />
                  <span className="text-xs text-foreground-secondary truncate">{file.name}</span>
                  <span className="text-xs text-foreground-muted">
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <button
                  onClick={() => handleRemovePending(index)}
                  disabled={isUploading}
                  className="p-1 hover:bg-border rounded transition-colors disabled:opacity-50"
                >
                  <svg className="w-3 h-3 text-foreground-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>

          {/* Upload button */}
          <button
            onClick={handleUploadClick}
            disabled={isUploading || disabled}
            className="w-full px-4 py-2 rounded-lg bg-accent text-white text-xs font-medium hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isUploading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                {parseProgress ? `Parsing ${parseProgress.filename}...` : 'Uploading...'}
              </span>
            ) : (
              `Upload & Parse ${pendingFiles.length} file${pendingFiles.length > 1 ? 's' : ''}`
            )}
          </button>
        </div>
      )}

      {/* Parse progress */}
      {isUploading && parseProgress && (
        <div className={`px-3 py-2 rounded-lg bg-accent/10 text-accent ${pendingFiles.length > 0 || !canAddMore ? 'mt-3' : ''}`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs">
              {parseProgress.status === 'docling' ? 'Parsing with Docling' :
               parseProgress.status === 'reading_text' ? 'Reading text file' :
               parseProgress.status === 'complete' ? 'Complete' :
               parseProgress.status === 'error' ? 'Error' : 'Processing'}
            </span>
            <span className="text-xs">{parseProgress.current}/{parseProgress.total}</span>
          </div>
          <div className="w-full h-1 bg-accent/20 rounded-full overflow-hidden">
            <div
              className="h-full bg-accent transition-all duration-300"
              style={{ width: `${(parseProgress.current / parseProgress.total) * 100}%` }}
            />
          </div>
        </div>
      )}
    </>
  );
}
