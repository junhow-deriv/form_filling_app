'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import type { Components } from 'react-markdown';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  // Custom component styling for markdown elements
  const components: Components = {
    // Paragraphs
    p: ({ children }) => <p className="mb-2 last:mb-0 break-words">{children}</p>,
    
    // Headings
    h1: ({ children }) => <h1 className="text-xl font-semibold mb-2 mt-3 first:mt-0">{children}</h1>,
    h2: ({ children }) => <h2 className="text-lg font-semibold mb-2 mt-3 first:mt-0">{children}</h2>,
    h3: ({ children }) => <h3 className="text-base font-semibold mb-2 mt-2 first:mt-0">{children}</h3>,
    h4: ({ children }) => <h4 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h4>,
    h5: ({ children }) => <h5 className="text-sm font-semibold mb-1 mt-2 first:mt-0">{children}</h5>,
    h6: ({ children }) => <h6 className="text-xs font-semibold mb-1 mt-2 first:mt-0">{children}</h6>,
    
    // Links - Open in new tab with security attributes
    a: ({ href, children }) => (
      <a 
        href={href} 
        className="text-accent hover:underline" 
        target="_blank" 
        rel="noopener noreferrer"
      >
        {children}
      </a>
    ),
    
    // Lists
    ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
    ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
    li: ({ children }) => <li className="ml-2">{children}</li>,
    
    // Strong/Bold
    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
    
    // Emphasis/Italic
    em: ({ children }) => <em className="italic">{children}</em>,
    
    // Blockquotes
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-current pl-3 italic opacity-80 my-2">
        {children}
      </blockquote>
    ),
    
    // Horizontal rule
    hr: () => <hr className="my-3 border-current opacity-20" />,
    
    // Inline code
    code: ({ children }) => (
      <code className="bg-black/10 px-1.5 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    ),
    
    // Tables (from remark-gfm)
    table: ({ children }) => (
      <div className="overflow-x-auto my-2">
        <table className="min-w-full border-collapse border border-current/20">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-black/5">{children}</thead>,
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => <tr className="border-b border-current/20">{children}</tr>,
    th: ({ children }) => (
      <th className="border border-current/20 px-3 py-2 text-left font-semibold">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border border-current/20 px-3 py-2">
        {children}
      </td>
    ),
  };

  return (
    <div className="text-sm markdown-content break-words whitespace-normal">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
