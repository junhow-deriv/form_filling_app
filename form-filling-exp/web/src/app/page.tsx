'use client';

import { useState, useCallback, useEffect } from 'react';
import { ChatMessage, FormField, PdfDisplayMode, StreamEvent, AgentLogEntry } from '@/types';
import { analyzePdf, streamAgentFill, hexToBytes, streamParseFiles, getSessionInfo, getSessionPdf, getSessionOriginalPdf } from '@/lib/api';
import { ContextFile, ParseProgress } from '@/components/ContextFilesUpload';
import {
  createMessage,
  getSessionIdFromUrl,
  setSessionIdInUrl,
} from '@/lib/session';
import LeftPanel from '@/components/LeftPanel';
import ChatPanel from '@/components/ChatPanel';
// Helper to generate unique IDs
const generateId = () => crypto.randomUUID();

export default function Home() {
  // API keys (handled by backend env vars)
  const [anthropicApiKey] = useState<string | null>(null);

  const [sessionId, setSessionId] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [fields, setFields] = useState<FormField[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [originalPdfBytes, setOriginalPdfBytes] = useState<Uint8Array | null>(null);  // For restored sessions
  const [filledPdfBytes, setFilledPdfBytes] = useState<Uint8Array | null>(null);
  const [pdfDisplayMode, setPdfDisplayMode] = useState<PdfDisplayMode>('original');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  // Track applied edits for multi-turn conversations
  const [appliedEdits, setAppliedEdits] = useState<Record<string, unknown> | null>(null);
  // Track agent session ID for resuming conversations (Claude SDK session)
  const [agentSessionId, setAgentSessionId] = useState<string | null>(null);
  // Track user session ID for backend state isolation (concurrent user support)
  const [userSessionId, setUserSessionId] = useState<string | null>(null);
  // Context files for the agent
  const [contextFiles, setContextFiles] = useState<ContextFile[]>([]);
  const [isUploadingContext, setIsUploadingContext] = useState(false);
  const [parseProgress, setParseProgress] = useState<ParseProgress | null>(null);
  // Track if first agent turn was successful (for continuation logic)
  const [hasSuccessfulFirstTurn, setHasSuccessfulFirstTurn] = useState(false);

  // Initialize session from URL or create new one
  useEffect(() => {
    const urlSessionId = getSessionIdFromUrl();

    if (urlSessionId) {
      // Use existing session ID from URL for both frontend and backend
      // This is the unified session ID that maps to form_states.session_id
      setSessionId(urlSessionId);
      setUserSessionId(urlSessionId);
      
      // Fetch session data from backend
      getSessionInfo(urlSessionId).then(async (sessionData) => {
        if (sessionData) {
          console.log('[Session Restore] Found session data:', sessionData);
          
          // Restore session state
          setAgentSessionId(sessionData.agent_session_id);
          setAppliedEdits(sessionData.applied_edits);
          setFields(sessionData.fields || []);

          const transformedMessages = (sessionData.messages || []).map(msg => ({
            ...msg,
            id: msg.id || generateId(),
            timestamp: typeof msg.timestamp === 'string' ? new Date(msg.timestamp) : msg.timestamp,
            status: msg.status as ChatMessage['status'] | undefined,
            toolCalls: msg.toolCalls || [],
            agentLog: msg.agentLog || [],
          }));
          setMessages(transformedMessages);
          
          setContextFiles(sessionData.context_files || []);
          
          if (sessionData.has_filled_pdf) {
            setHasSuccessfulFirstTurn(true);
          }
          
          // Fetch PDFs in parallel
          const promises: Promise<void>[] = [];
          
          if (sessionData.has_filled_pdf) {
            promises.push(
              getSessionPdf(urlSessionId).then(bytes => {
                if (bytes) {
                  console.log('[Session Restore] Restored filled PDF:', bytes.length, 'bytes');
                  setFilledPdfBytes(bytes);
                  setPdfDisplayMode('filled');
                  
                  // Create File object from filled PDF bytes
                  if (sessionData.pdf_filename) {
                    const file = new File([bytes as unknown as Blob], sessionData.pdf_filename, { type: 'application/pdf' });
                    setFile(file);
                  }
                }
              })
            );
          }
          
          if (sessionData.has_original_pdf) {
            promises.push(
              getSessionOriginalPdf(urlSessionId).then(bytes => {
                if (bytes) {
                  console.log('[Session Restore] Restored original PDF:', bytes.length, 'bytes');
                  setOriginalPdfBytes(bytes);
                }
              })
            );
          }
          
          await Promise.all(promises);
          console.log('[Session Restore] Session fully restored');
        } else {
          console.log('[Session Restore] No session data found for:', urlSessionId);
        }
      }).catch(err => {
        console.error('[Session Restore] Failed to restore session:', err);
      });
    } else {
      const unifiedSessionId = generateId();
      setSessionId(unifiedSessionId);
      setUserSessionId(unifiedSessionId);
      setSessionIdInUrl(unifiedSessionId);
    }
  }, []);

  const handleNewForm = useCallback(() => {
    const newSessionId = generateId();
    window.location.href = `/?session=${newSessionId}`;
  }, []);

  // Handle file selection and analysis
  const handleFileSelect = useCallback(async (selectedFile: File | null) => {
    if (!selectedFile) {
      setFile(null);
      setFields([]);
      setOriginalPdfBytes(null);  // Clear restored original PDF
      setFilledPdfBytes(null);
      setPdfDisplayMode('original');
      setAppliedEdits(null);  // Clear edits when resetting
      setAgentSessionId(null);  // Clear agent session when resetting
      setUserSessionId(null);  // Clear user session when resetting
      setContextFiles([]);  // Clear context files when resetting
      return;
    }

    setFile(selectedFile);
    setOriginalPdfBytes(null);  // Clear restored original PDF for new file
    setFilledPdfBytes(null);
    setPdfDisplayMode('original');
    setAppliedEdits(null);  // Clear edits for new file
    setAgentSessionId(null);  // Clear agent session for new file
    setUserSessionId(null);  // Clear user session for new file
    setContextFiles([]);  // Clear context files for new file
    setIsAnalyzing(true);

    try {
      for await (const event of analyzePdf(selectedFile, sessionId)) {
        if (event.type === 'fields_detected' && event.fields) {
          setFields(event.fields);
        }
        
        if (event.type === 'system_message' && event.content) {
          setMessages((prev) => [
            ...prev,
            createMessage('system', event.content || ''),
          ]);
        }
        
        if (event.type === 'error' && event.error) {
          setMessages((prev) => [
            ...prev,
            createMessage('system', `Error: ${event.error}`),
          ]);
        }
      }
      
      setUserSessionId(sessionId);
    } catch (error) {
      console.error('Analysis error:', error);
      setMessages((prev) => [
        ...prev,
        createMessage('system', `Error analyzing PDF: ${error instanceof Error ? error.message : 'Unknown error'}`),
      ]);
    } finally {
      setIsAnalyzing(false);
    }
  }, [sessionId]);

  // Handle parsing context files
  const handleParseFiles = useCallback(
    async (files: File[]) => {
      setIsUploadingContext(true);
      setParseProgress(null);

      // Use existing session ID (already unified with backend)
      // If for some reason it doesn't exist, use the frontend sessionId
      let currentUserSessionId = userSessionId;
      if (!currentUserSessionId) {
        currentUserSessionId = sessionId;
        setUserSessionId(currentUserSessionId);
      }

      try {
        const results: ContextFile[] = [];

        for await (const event of streamParseFiles(files, currentUserSessionId)) {
          if (event.type === 'progress' && event.current !== undefined && event.total !== undefined && event.filename && event.status) {
            setParseProgress({
              current: event.current,
              total: event.total,
              filename: event.filename,
              status: event.status,
              error: event.error,
            });
          }

          if (event.type === 'system_message' && event.content) {
            setMessages((prev) => [
              ...prev,
              createMessage('system', event.content || ''),
            ]);
          }

          if (event.type === 'error' && event.error) {
            // Show error message to user
            setMessages((prev) => [
              ...prev,
              createMessage('system', `Error: ${event.error}`),
            ]);
            return; // Stop processing
          }

          if (event.type === 'complete' && event.results) {
            for (const result of event.results) {
              if (result.success && result.document_id) {
                results.push({
                  filename: result.filename,
                  document_id: result.document_id,
                });
              }
            }
          }
        }

        // Add new files to existing context files
        setContextFiles((prev) => [...prev, ...results]);
      } catch (error) {
        console.error('Parse files error:', error);
        setMessages((prev) => [
          ...prev,
          createMessage('system', `Error parsing files: ${error instanceof Error ? error.message : 'Unknown error'}`),
        ]);
      } finally {
        setIsUploadingContext(false);
        setParseProgress(null);
      }
    },
    [userSessionId, sessionId]
  );

  // Handle sending a chat message
  const handleSendMessage = useCallback(
    async (content: string) => {
      if (!file) {
        setMessages((prev) => [
          ...prev,
          createMessage('system', 'Please upload a PDF first'),
        ]);
        return;
      }

      // Add user message
      const userMessage = createMessage('user', content);
      setMessages((prev) => [...prev, userMessage]);

      // Create assistant message placeholder with empty agent log
      const assistantMessage: ChatMessage = {
        ...createMessage('assistant', '', 'streaming'),
        agentLog: [],
      };
      setMessages((prev) => [...prev, assistantMessage]);

      setIsProcessing(true);
      setStatusMessage('Starting agent...');

      // Determine if this is a continuation (we have a previous successful agent turn)
      const isContinuation = Boolean(agentSessionId && filledPdfBytes && hasSuccessfulFirstTurn);

      let finalContent = '';
      let appliedCount = 0;
      let newAppliedEdits: Record<string, unknown> | null = null;
      let newAgentSessionId: string | null = null;
      let newFilledPdfBytes: Uint8Array | null = null;

      try {
        for await (const event of streamAgentFill({
          file,
          instructions: content,
          filledPdfBytes: isContinuation ? filledPdfBytes : null,
          isContinuation,
          previousEdits: appliedEdits,
          resumeSessionId: agentSessionId,
          userSessionId: userSessionId,
          anthropicApiKey: anthropicApiKey,
          contextFileIds: contextFiles.length > 0 ? contextFiles : undefined,
        })) {
          const logEntry = createLogEntry(event);

          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantMessage.id) return m;

              const updatedLog = logEntry
                ? [...(m.agentLog || []), logEntry]
                : m.agentLog;

              // Update status message for UI
              if (logEntry) {
                setStatusMessage(logEntry.content);
              }

              return {
                ...m,
                agentLog: updatedLog,
              };
            })
          );

          // Handle special events
          if (event.type === 'fields_detected' && event.fields) {
            setFields(event.fields);
            setStatusMessage(`Detected ${event.field_count || event.fields.length} form fields`);
          }

          if (event.type === 'complete') {
            appliedCount = event.applied_count || 0;
            // Track all applied edits for multi-turn
            if (event.applied_edits) {
              newAppliedEdits = event.applied_edits;
            }
            // Track agent session ID for resuming conversations
            if (event.session_id) {
              newAgentSessionId = event.session_id;
            }
            // Use final_message from backend if available, otherwise construct fallback
            if (event.final_message) {
              finalContent = event.final_message;
            } else {
              const totalEdits = newAppliedEdits ? Object.keys(newAppliedEdits).length : appliedCount;
              if (isContinuation) {
                finalContent = `Updated ${appliedCount} fields. Total: ${totalEdits} fields filled.`;
              } else {
                finalContent = `Successfully filled ${appliedCount} form fields.`;
              }
            }
          }

          if (event.type === 'pdf_ready' && event.pdf_bytes) {
            const bytes = hexToBytes(event.pdf_bytes);
            newFilledPdfBytes = bytes;
            setFilledPdfBytes(bytes);
            setPdfDisplayMode('filled');
            setStatusMessage('PDF filled successfully!');
            
            // Mark first turn as successful
            if (!hasSuccessfulFirstTurn) {
              setHasSuccessfulFirstTurn(true);
            }
          }

          if (event.type === 'error') {
            finalContent = event.error || 'An error occurred';
          }
        }

        // Update applied edits after successful completion
        if (newAppliedEdits) {
          setAppliedEdits(newAppliedEdits);
        }

        // Update agent session ID for multi-turn conversations
        if (newAgentSessionId) {
          setAgentSessionId(newAgentSessionId);
        }

        // Mark assistant message as complete
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessage.id
              ? {
                  ...m,
                  status: 'complete',
                  content: finalContent || `Filled ${appliedCount} fields.`,
                }
              : m
          )
        );
      } catch (error) {
        console.error('Agent error:', error);
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessage.id
              ? {
                  ...m,
                  status: 'error',
                  content: `Error: ${errorMessage}`,
                  agentLog: [
                    ...(m.agentLog || []),
                    {
                      id: generateId(),
                      type: 'error' as const,
                      timestamp: new Date(),
                      content: errorMessage,
                    },
                  ],
                }
              : m
          )
        );
      } finally {
        setIsProcessing(false);
        setStatusMessage('');
      }
    },
    [file, filledPdfBytes, appliedEdits, agentSessionId, userSessionId, sessionId, fields, messages, anthropicApiKey, contextFiles]
  );

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="flex-shrink-0 px-6 py-3 border-b border-border flex items-center justify-end">
        <div className="flex items-center gap-4">
          {sessionId && (
            <div className="text-xs text-foreground-muted">
              Session: {sessionId.slice(0, 8)}...
            </div>
          )}
          <a
            href="/docs"
            target="_blank"
            className="text-xs text-foreground-muted hover:text-accent transition-colors"
          >
            API Docs
          </a>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left panel - PDF upload and preview */}
        <div className="w-1/2 border-r border-border flex flex-col overflow-hidden">
          <LeftPanel
            file={file}
            onFileSelect={handleFileSelect}
            onNewForm={handleNewForm}
            fields={fields}
            originalPdfBytes={originalPdfBytes}
            filledPdfBytes={filledPdfBytes}
            pdfDisplayMode={pdfDisplayMode}
            onPdfDisplayModeChange={setPdfDisplayMode}
            isAnalyzing={isAnalyzing}
            isProcessing={isProcessing}
          />
        </div>

        {/* Right panel - Chat interface */}
        <div className="w-1/2 flex flex-col overflow-hidden">
          <ChatPanel
            messages={messages}
            onSendMessage={handleSendMessage}
            isProcessing={isProcessing}
            disabled={!file || fields.length === 0}
            statusMessage={statusMessage}
            contextFiles={contextFiles}
            onContextFilesChange={setContextFiles}
            onParseFiles={handleParseFiles}
            isUploadingContext={isUploadingContext}
            parseProgress={parseProgress}
            shouldDisableContextUpload={hasSuccessfulFirstTurn}
          />
        </div>
      </main>
    </div>
  );
}

// Create a log entry from a stream event
function createLogEntry(event: StreamEvent): AgentLogEntry | null {
  const id = generateId();
  const timestamp = new Date();

  switch (event.type) {
    case 'init':
      return {
        id,
        type: 'status',
        timestamp,
        content: event.message || 'Initializing agent...',
      };

    case 'status':
      return {
        id,
        type: 'status',
        timestamp,
        content: event.message || 'Processing...',
      };

    case 'tool_use':
      if (event.friendly && event.friendly.length > 0) {
        // Clean up markdown formatting
        const cleanedActions = event.friendly.map((f) => f.replace(/\*\*/g, ''));

        if (event.friendly.length > 1) {
          return {
            id,
            type: 'tool_call',
            timestamp,
            content: `Filling ${event.friendly.length} fields`,
            details: cleanedActions.join(', '),
          };
        } else {
          return {
            id,
            type: 'tool_call',
            timestamp,
            content: cleanedActions[0],
          };
        }
      }
      return null;

    case 'user':
      // Tool results - event.friendly is string[] from StreamEvent
      if (event.friendly && event.friendly.length > 0) {
        return {
          id,
          type: 'tool_result',
          timestamp,
          content: event.friendly.join(', '),
        };
      }
      return null;

    case 'assistant':
      if (event.text) {
        return {
          id,
          type: 'thinking',
          timestamp,
          content: 'Agent thinking...',
          details: event.text.slice(0, 100) + (event.text.length > 100 ? '...' : ''),
        };
      }
      return null;

    case 'complete':
      return {
        id,
        type: 'complete',
        timestamp,
        content: `Completed - filled ${event.applied_count || 0} fields`,
      };

    case 'error':
      return {
        id,
        type: 'error',
        timestamp,
        content: event.error || 'An error occurred',
      };

    case 'pdf_ready':
      return {
        id,
        type: 'complete',
        timestamp,
        content: 'Form filled successfully',
      };

    default:
      return null;
  }
}
