import { useState, useRef, useEffect } from 'react';
import { API_ENDPOINTS } from '@/utils/api';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatBoxProps {
  signName: string;
  onClose: () => void;
  onMinimize?: () => void;
}

const ChatBox = ({ signName, onClose, onMinimize }: ChatBoxProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize chat on component mount
  useEffect(() => {
    const initializeChat = async () => {
      setIsLoading(true);
      try {
        console.log('Initializing chat for sign:', signName);
        
        const response = await fetch(`${API_ENDPOINTS.chat}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            sign_name: signName,
            message: '',
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to initialize chat: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Chat initialized successfully:', data);
        
        setSessionId(data.session_id);
        
        // Check if we have a valid response
        if (!data.response) {
          console.error('Empty greeting response received:', data);
          throw new Error('Empty greeting received');
        }
        
        setMessages([{ role: 'assistant', content: data.response }]);
      } catch (error) {
        console.error('Error initializing chat:', error);
        // Add fallback message if initialization fails
        setMessages([
          {
            role: 'assistant',
            content: `Hi! I'm your ASL tutor for the letter '${signName}'. What would you like to know about this sign?`,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    initializeChat();
  }, [signName]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    
    // Add user message immediately
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    
    // Show loading state
    setIsLoading(true);

    try {
      console.log('Sending message to backend:', {
        sign_name: signName,
        message: userMessage,
        session_id: sessionId,
      });
      
      const response = await fetch(`${API_ENDPOINTS.chat}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sign_name: signName,
          message: userMessage,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to send message: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Received response from backend:', data);
      
      // Update session ID if needed
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }
      
      // Validate response content
      if (!data.response || data.response.trim() === "") {
        console.error('Empty response received from backend:', data);
        // Add a fallback error message
        setMessages((prev) => [
          ...prev,
          { 
            role: 'assistant', 
            content: `I'm having trouble generating a response about the '${signName}' sign. Could you try asking another question?` 
          },
        ]);
      } else {
        // Add assistant response
        setMessages((prev) => [...prev, { role: 'assistant', content: data.response }]);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages((prev) => [
        ...prev,
        { 
          role: 'assistant', 
          content: `Sorry, I had trouble responding. Error: ${error instanceof Error ? error.message : 'Unknown error'}` 
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center p-2 bg-blue-600 text-white">
        <h3 className="font-bold text-sm">ASL Tutor - Letter '{signName}'</h3>
        <div className="flex space-x-1">
          {onMinimize && (
            <button 
              onClick={onMinimize}
              className="text-white hover:bg-blue-700 rounded-full p-1"
              aria-label="Minimize chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 12H6" />
              </svg>
            </button>
          )}
          <button 
            onClick={onClose}
            className="text-white hover:bg-blue-700 rounded-full p-1"
            aria-label="Close chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 p-3 overflow-y-auto bg-gray-50" style={{ maxHeight: "calc(100% - 80px)" }}>
        {messages.length === 0 && isLoading ? (
          <div className="flex justify-center items-center h-full">
            <div className="animate-pulse text-gray-500 text-sm">Loading...</div>
          </div>
        ) : (
          <div className="space-y-3">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-2 text-sm ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-800'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSendMessage} className="p-2 border-t">
        <div className="flex gap-1">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about this sign..."
            disabled={isLoading}
            className="flex-1 px-3 py-1 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading || !inputValue.trim()}
            className="px-3 py-1 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {isLoading ? (
              <span className="flex items-center">
                <svg className="animate-spin -ml-1 mr-1 h-3 w-3 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Send
              </span>
            ) : (
              'Send'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatBox; 