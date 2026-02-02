'use client';

import { useCallback } from 'react';
import FileUploadZone, { ParseProgress } from './FileUploadZone';

export interface ContextFile {
  filename: string;
  document_id: string;
}


interface ContextFilesUploadProps {
  files: ContextFile[];
  onFilesChange: (files: ContextFile[]) => void;
  onParseFiles: (files: File[]) => Promise<void>;
  isUploading: boolean;
  parseProgress: ParseProgress | null;
  disabled?: boolean;
  maxFiles?: number;
}

export default function ContextFilesUpload({
  files,
  onFilesChange,
  onParseFiles,
  isUploading,
  parseProgress,
  disabled,
  maxFiles = 10,
}: ContextFilesUploadProps) {
  const handleRemoveUploaded = useCallback((index: number) => {
    onFilesChange(files.filter((_, i) => i !== index));
  }, [files, onFilesChange]);

  const totalCount = files.length;

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-foreground-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="text-xs font-medium text-foreground-secondary">
            Context Files ({totalCount}/{maxFiles})
          </span>
        </div>
      </div>

      {/* File Upload Zone */}
      <FileUploadZone
        onUpload={onParseFiles}
        isUploading={isUploading}
        parseProgress={parseProgress}
        disabled={disabled}
        maxFiles={maxFiles}
        currentFileCount={files.length}
        dropZoneSize="small"
        inputId="context-files-input"
      />

      {/* Uploaded files list */}
      {files.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-foreground-muted">Uploaded:</p>
          {files.map((file, index) => (
            <div
              key={`uploaded-${index}`}
              className="flex items-center justify-between px-3 py-2 rounded-lg bg-success/10"
            >
              <div className="flex items-center gap-2 min-w-0">
                <svg className="w-4 h-4 text-success flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-xs text-foreground-secondary truncate">{file.filename}</span>
              </div>
              <button
                onClick={() => handleRemoveUploaded(index)}
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
      )}
    </div>
  );
}
