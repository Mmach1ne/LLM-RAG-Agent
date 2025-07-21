import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './index.css';

const API_URL = 'http://localhost:8000';

// Message type for chat messages
type Message = {
  from: 'user' | 'agent';
  text: string;
  widget?: any;
};

// Bot architecture types
type MemorySystem = {
  database: string;
  storage: string;
  conversations: number;
  recallTime: string;
};

type Skill = {
  name: string;
  subtitle: string;
  icon: string;
};

type Task = {
  id: string;
  name: string;
  status: 'active' | 'processing' | 'complete';
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

  // Bot architecture state - with real data
  const [memorySystem, setMemorySystem] = useState<MemorySystem>({
    database: 'SQLite',
    storage: '0MB',
    conversations: 0,
    recallTime: '0ms'
  });

  const [skills, setSkills] = useState<Skill[]>([
    { name: 'Math Solver', subtitle: '0 calls', icon: 'λ' },
    { name: 'Code Generator', subtitle: '0 calls', icon: '</>' },
    { name: 'Data Analysis', subtitle: '0 calls', icon: '📊' },
    { name: 'Web Search', subtitle: '0 calls', icon: '🔍' },
    { name: 'Context Memory', subtitle: 'Always On', icon: '💭' },
    { name: 'Add Skill', subtitle: 'Modular', icon: '+' }
  ]);

  const [tasks, setTasks] = useState<Task[]>([]);
  const [responseTime, setResponseTime] = useState('~0ms');
  const [botState, setBotState] = useState('idle');
  const [taskCount, setTaskCount] = useState(0);
  const [memoryCount, setMemoryCount] = useState(0);

  // Calculate storage size based on messages
  const calculateStorage = () => {
    const totalChars = messages.reduce((acc, msg) => acc + msg.text.length, 0);
    const sizeInBytes = totalChars * 2; // Rough estimate: 2 bytes per character
    if (sizeInBytes < 1024) return `${sizeInBytes}B`;
    if (sizeInBytes < 1024 * 1024) return `${(sizeInBytes / 1024).toFixed(1)}KB`;
    return `${(sizeInBytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  // Update storage when messages change
  useEffect(() => {
    setMemorySystem(prev => ({
      ...prev,
      storage: calculateStorage(),
      conversations: messages.filter(m => m.from === 'user').length
    }));}, [messages]);
  // scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Track message processing time
  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;
    
    const startTime = Date.now();
    setMessages(m => [...m, { from: 'user', text }]);
    setInput('');
    setIsTyping(true);
    setBotState('processing');

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
      
      // Calculate actual response time
      const endTime = Date.now();
      const actualResponseTime = endTime - startTime;
      setResponseTime(`~${actualResponseTime}ms`);
      
      setMessages(m => [...m, { from: 'agent', text: botText, widget }]);
      
      // Update conversation count
      setMemorySystem(prev => ({
        ...prev,
        conversations: prev.conversations + 1
      }));
    } catch (error) {
      console.error('Error:', error);
      setMessages(m => [...m, { from: 'agent', text: 'Sorry, I encountered an error.' }]);
    } finally {
      setIsTyping(false);
      setBotState('idle');
    }
  };

  // Fetch real status data
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/api/status`);
        const data = await res.json();
        
        // Update memory system stats
        if (data.memory) {
          setMemorySystem({
            database: data.memory.database || 'SQLite',
            storage: data.memory.storage || '0MB',
            conversations: data.memory.conversations || messages.length,
            recallTime: data.memory.recallTime || '0ms'
          });
          setMemoryCount(data.memory.count || 0);
        }
        
        // Update skills with real call counts
        if (data.skills) {
          setSkills(prevSkills => prevSkills.map(skill => {
            const skillData = data.skills[skill.name.toLowerCase().replace(/\s+/g, '_')];
            return {
              ...skill,
              subtitle: skillData ? `${skillData.calls || 0} calls` : skill.subtitle
            };
          }));
        }
        
        // Update bot state
        if (data.state) {
          setBotState(data.state);
        }
        
        // Update task count
        if (data.tasks) {
          setTaskCount(data.tasks.active || 0);
          
          // Update task queue with real tasks
          if (data.tasks.queue && data.tasks.queue.length > 0) {
            setTasks(data.tasks.queue.map((task: any, idx: number) => ({
              id: `#${1000 + idx}`,
              name: task.name || 'Task',
              status: task.status || 'active'
            })));
          }
        }
        
        // Update response time
        if (data.performance) {
          setResponseTime(`~${data.performance.avgResponseTime || 0}ms`);
        }
      } catch (error) {
        console.error('Error fetching status:', error);
      }
    };

    // Fetch immediately and then every 5 seconds
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    
    return () => clearInterval(interval);
  }, [messages.length]);

  return (
    <div className="app-bg">
      {/* Bot Architecture Sidebar */}
      <aside className="bot-architecture">
        <div className="architecture-header">
          <h3>BOT ARCHITECTURE</h3>
          <div className="status-dot"></div>
        </div>

        {/* Memory System */}
        <div className="memory-section">
          <h4>MEMORY SYSTEM</h4>
          <div className="memory-stats">
            <div className="stat-row">
              <span className="stat-label">DATABASE</span>
              <span className="stat-value">{memorySystem.database}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">STORAGE</span>
              <span className="stat-value">{memorySystem.storage}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">CONVERSATIONS</span>
              <span className="stat-value">{memorySystem.conversations}</span>
            </div>
            <div className="stat-row">
              <span className="stat-label">RECALL TIME</span>
              <span className="stat-value">{memorySystem.recallTime}</span>
            </div>
          </div>
        </div>

        {/* Active Skills */}
        <div className="skills-section">
          <h4>ACTIVE SKILLS</h4>
          <div className="skills-grid">
            {skills.map((skill, idx) => (
              <div key={idx} className="skill-card">
                <div className="skill-icon">{skill.icon}</div>
                <div className="skill-name">{skill.name}</div>
                <div className="skill-subtitle">{skill.subtitle}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Task Queue */}
        <div className="task-section">
          <h4>TASK QUEUE</h4>
          <div className="task-list">
            {tasks.map((task) => (
              <div key={task.id} className={`task-item task-${task.status}`}>
                <span className="task-id">{task.id}</span>
                <span className="task-name">{task.name}</span>
                <div className={`task-indicator ${task.status}`}></div>
              </div>
            ))}
          </div>
          <div className="response-time">
            <span>Response Time:</span>
            <span className="time-value">{responseTime}</span>
          </div>
        </div>
      </aside>

      {/* Chat Container */}
      <div className="chat-container">
        {/* header */}
        <aside className="w-full p-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-t-2xl">
          <h2 className="text-3xl font-bold text-white mb-4">RayBot</h2>
          <div className="flex justify-center gap-4">
            <div className="status-item">
              <div className="text-sm uppercase">State</div>
              <div className="font-semibold">{botState}</div>
            </div>
            <div className="status-item">
              <div className="text-sm uppercase">Tasks</div>
              <div className="font-semibold">{taskCount}</div>
            </div>
            <div className="status-item">
              <div className="text-sm uppercase">Memories</div>
              <div className="font-semibold">{memoryCount}</div>
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