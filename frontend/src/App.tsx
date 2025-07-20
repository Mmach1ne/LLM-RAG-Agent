import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './index.css';            // your global styles

const API_URL = 'http://34.83.90.220:8000';  // Updated API URL

// Message type for chat messages
type Message = {
  from: 'user' | 'agent';
  text: string;
  widget?: any;
};

// WeatherWidget component
const WeatherWidget = ({ widget }: { widget: any }) => (
  <div className="weather-widget p-4 rounded-xl flex flex-col items-center mb-2">
    <div className="text-lg font-bold mb-1">{widget.location}</div>
    {widget.icon && (
      <img src={widget.icon} alt={widget.condition} className="w-16 h-16 mb-1" />
    )}
    <div className="text-4xl font-extrabold mb-1">{widget.temperature}</div>
    <div className="text-md mb-2">{widget.condition}</div>
    <div className="text-xs text-gray-500">Weather Widget</div>
  </div>
);

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // send message handler
  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    setMessages(m => [...m, { from: 'user', text }]);
    setInput('');
    setIsTyping(true);

    try {
      const res = await fetch(`${API_URL}/api/process`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify({ input: text }),
      });
      const { response: botText, widget } = await res.json();
      setMessages(m => [...m, { from: 'agent', text: botText, widget }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(m => [...m, { from: 'agent', text: 'Sorry, I encountered an error.' }]);
    } finally {
      setIsTyping(false);
    }
  };

  // Add a useEffect to fetch initial status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        // Ping status endpoint (no data needed here)
        await fetch(`${API_URL}/api/status`);
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    fetchStatus();
  }, []);

  return (
    <div className="app-bg">
      <div className="chat-container">
        {/* header */}
        <aside className="w-full p-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-t-2xl">
          <h2 className="text-3xl font-bold text-white mb-4">RayBot</h2>
          <div className="flex justify-center gap-4">
            <div className="status-item">
              <div className="text-sm uppercase">State</div>
              <div className="font-semibold">idle</div>
            </div>
            <div className="status-item">
              <div className="text-sm uppercase">Tasks</div>
              <div className="font-semibold">0</div>
            </div>
            <div className="status-item">
              <div className="text-sm uppercase">Memories</div>
              <div className="font-semibold">0</div>
            </div>
          </div>
        </aside>

        {/* messages */}
        <main className="flex-1 flex flex-col justify-between">
          <div className="messages">
            <AnimatePresence>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={`chat-bubble ${
                    msg.from === 'user' ? 'user-bubble ml-auto' : 'agent-bubble mr-auto'
                  }`}
                >
                  {msg.widget && msg.widget.type === 'weather' ? (
                    <WeatherWidget widget={msg.widget} />
                  ) : (
                    msg.text
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {isTyping && (
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* input area */}
          <div className="input-area">
            <motion.input
              whileFocus={{ scale: 1.01 }}
              type="text"
              className="flex-1 bg-[#1f2937] text-white placeholder-gray-400 border border-[#334155] rounded-xl p-4 focus:outline-none"
              placeholder="Type your message..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
            />
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={sendMessage}
              disabled={!input.trim()}
              className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-6 py-4 rounded-xl font-semibold disabled:opacity-50"
            >
              Send
            </motion.button>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
