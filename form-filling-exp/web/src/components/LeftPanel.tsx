'use client';

import { useRef, useState } from 'react';
import { FormField, PdfDisplayMode } from '@/types';
import PdfUpload from './PdfUpload';
import FormFields from './FormFields';
import PdfViewer from './PdfViewer';
import { detectFields, downloadPdf } from '@/lib/api';

interface LeftPanelProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  onNewForm: () => void;
  fields: FormField[];
  originalPdfBytes: Uint8Array | null;  // For restored sessions
  filledPdfBytes: Uint8Array | null;
  pdfDisplayMode: PdfDisplayMode;
  onPdfDisplayModeChange: (mode: PdfDisplayMode) => void;
  isAnalyzing: boolean;
  isProcessing: boolean;
}

export default function LeftPanel({
  file,
  onFileSelect,
  onNewForm,
  fields,
  originalPdfBytes,
  filledPdfBytes,
  pdfDisplayMode,
  onPdfDisplayModeChange,
  isAnalyzing,
  isProcessing,
}: LeftPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFileForDetection, setSelectedFileForDetection] = useState<File | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);

  const handleDetectFieldsClick = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (selectedFileForDetection) {
      // File is selected - run detection
      setIsDetecting(true);
      try {
        console.log('Running field detection on:', selectedFileForDetection.name);
        const pdfBytes = await detectFields(selectedFileForDetection);
        
        // Automatically download the interactive PDF
        const filename = selectedFileForDetection.name.replace('.pdf', '_interactive.pdf');
        downloadPdf(pdfBytes, filename);
        
        console.log('Field detection completed! Interactive PDF downloaded.');
        // Clear selection after successful detection
        setSelectedFileForDetection(null);
      } catch (error) {
        console.error('Field detection error:', error);
        alert(`Error: ${error instanceof Error ? error.message : 'Failed to detect fields'}`);
      } finally {
        setIsDetecting(false);
      }
    } else {
      // No file selected - open file picker
      fileInputRef.current?.click();
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const selectedFile = e.target.files[0];
      setSelectedFileForDetection(selectedFile);
      console.log('File selected for field detection:', selectedFile.name);
      
      // Reset the input so the same file can be selected again if needed
      e.target.value = '';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Hidden file input for field detection */}
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <h2 className="text-sm font-semibold">Form</h2>
            <p className="text-xs text-foreground-muted">
              {file ? file.name : 'Upload and preview your PDF'}
            </p>
          </div>
          
          {/* Detect Fields Button - Left side with some gap */}
          <button
            type="button"
            onClick={handleDetectFieldsClick}
            disabled={isProcessing || isDetecting}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDetecting ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Detecting...
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                {selectedFileForDetection ? 'Click to run detection' : 'Detect Fields'}
              </>
            )}
          </button>
          
          {/* Selected file indicator */}
          {selectedFileForDetection && (
            <div className="flex items-center gap-2 px-3 py-1 text-xs bg-background-tertiary rounded-md border border-border">
              <svg className="w-3.5 h-3.5 text-foreground-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
              <span className="text-foreground-secondary">{selectedFileForDetection.name}</span>
              <button
                type="button"
                onClick={() => setSelectedFileForDetection(null)}
                className="ml-1 text-foreground-muted hover:text-foreground-secondary"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
        </div>
        
        {/* New Form Button - Right side */}
        {file && (
          <button
            type="button"
            onClick={onNewForm}
            disabled={isProcessing}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md bg-background-tertiary text-foreground-muted hover:text-foreground-secondary hover:bg-border transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            New Form
          </button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col p-4 gap-4 overflow-hidden">
        {/* Upload section - only show when no file is selected AND no restored PDF bytes */}
        {!file && !originalPdfBytes && !filledPdfBytes && (
          <div className="flex-1 flex items-center justify-center">
            <div className="w-full max-w-md">
              <PdfUpload
                onFileSelect={onFileSelect}
                selectedFile={file}
                disabled={isProcessing}
              />
            </div>
          </div>
        )}

        {/* Loading state */}
        {isAnalyzing && (
          <div className="flex-1 flex items-center justify-center">
            <div className="flex items-center gap-3 text-foreground-muted">
              <div className="w-5 h-5 border-2 border-current border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">Analyzing PDF...</span>
            </div>
          </div>
        )}

        {/* PDF Viewer - takes full space when file is present OR we have restored PDF bytes */}
        {(file || originalPdfBytes || filledPdfBytes) && !isAnalyzing && (
          <PdfViewer
            originalFile={file}
            originalPdfBytes={originalPdfBytes}
            filledPdfBytes={filledPdfBytes}
            mode={pdfDisplayMode}
            onModeChange={onPdfDisplayModeChange}
          />
        )}

        {/* Fields panel - collapsible at bottom */}
        {fields.length > 0 && !isAnalyzing && (
          <details className="flex-shrink-0 border-t border-border pt-3">
            <summary className="cursor-pointer text-sm font-medium text-foreground-secondary hover:text-foreground transition-colors py-1 flex items-center gap-2 select-none">
              <svg className="w-4 h-4 transition-transform [details[open]>&]:rotate-90" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              Detected Fields ({fields.length})
            </summary>
            <div className="mt-2 max-h-[200px] overflow-y-auto">
              <FormFields fields={fields} />
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
